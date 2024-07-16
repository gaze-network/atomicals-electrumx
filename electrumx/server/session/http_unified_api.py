import asyncio
import base64
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

from aiohttp.web import json_response
from aiohttp.web_urldispatcher import UrlDispatcher
from aiorpcx import run_in_thread

from electrumx.lib import util
from electrumx.lib.hash import (
    HASHX_LEN,
    double_sha256,
    hash_to_hex_str,
    hex_str_to_hash,
    sha256,
)
from electrumx.lib.script2addr import (
    get_address_from_output_script,
    get_script_from_address,
)
from electrumx.lib.util_atomicals import (
    DFT_MINT_MAX_MAX_COUNT_DENSITY,
    compact_to_location_id_bytes,
    location_id_bytes_to_compact,
    parse_protocols_operations_from_witness_array,
)
from electrumx.server.db import UTXO, AtomicalUTXO
from electrumx.server.history import TXNUM_LEN

if TYPE_CHECKING:
    from aiohttp.web import Request, Response

    from electrumx.server.controller import SessionManager

supported_ops = ["dft", "mint-dft", "mint-ft", "split", "transfer", "burn", "custom-color"]

MAX_UINT64 = 9223372036854775807


class MAX_LIMIT:
    get_arc20_balances = 5000
    get_arc20_balances_batch_queries = 10
    get_arc20_transactions = 3000
    get_arc20_holders = 1000
    get_arc20_utxos = 3000  # large limit since pagination does not help performance
    get_arc20_token_list = 1000


class DEFAULT_LIMIT:
    get_arc20_balances = 100
    get_arc20_transactions = 100
    get_arc20_holders = 100
    get_arc20_utxos = 100
    get_arc20_token_list = 100


class JSONBytesEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, bytes):
            # use base64 encoding for bytes
            return base64.b64encode(obj).decode()
        return super().default(obj)


def format_response(result: "dict | None", status: "int | None" = None, error: "str | None" = None) -> "Response":
    if error:
        if status is None:
            status = 500
        return json_response(
            {
                "error": error,
            },
            status=status,
            dumps=lambda o: json.dumps(o, cls=JSONBytesEncoder),
        )
    if status is None:
        status = 200
    return json_response(
        {
            "error": None,
            "result": result,
        },
        status=status,
        dumps=lambda o: json.dumps(o, cls=JSONBytesEncoder),
    )


def scripthash_to_hashX(script_hash: bytes) -> "Optional[bytes]":
    if len(script_hash) == 32:
        return script_hash[:HASHX_LEN]
    return None


def get_decimals():
    return 0  # no decimal point for arc20


@dataclass
class BalanceQuery:
    address: str
    atomical_id: "bytes | None"
    block_height: "int | None"
    limit: "int"
    offset: "int"


class BadRequestException(Exception):
    pass


class HttpUnifiedAPIHandler(object):
    def __init__(
        self,
        session_mgr: "SessionManager",
    ):
        self.logger = util.class_logger(__name__, self.__class__.__name__)
        self.session_mgr = session_mgr

    def error_handler(func):
        async def wrapper(self: "HttpUnifiedAPIHandler", *args, **kwargs):
            try:
                return await func(self, *args, **kwargs)
            except BadRequestException as e:
                return format_response(None, 400, str(e))
            except Exception as e:
                self.logger.exception(f"Request has failed with exception: {repr(e)}")
                return format_response(None, 500, "Internal Server Error")

        return wrapper

    def mount_routes(self, router: UrlDispatcher):
        router.add_get("/v2/arc20/block", self.get_block_height)
        router.add_get("/v2/arc20/balances/wallet/{wallet}", self.get_arc20_balance)
        router.add_post("/v2/arc20/balances/wallet/batch", self.get_arc20_balances_batch)
        router.add_get("/v2/arc20/transactions", self.get_arc20_transactions)
        router.add_get("/v2/arc20/holders/{id}", self.get_arc20_holders)
        router.add_get("/v2/arc20/info/{id}", self.get_arc20_token)
        router.add_get("/v2/arc20/utxos/wallet/{wallet}", self.get_arc20_utxos)
        router.add_get("/v2/arc20/tokens", self.get_arc20_token_list)

    def _resolve_ticker_to_atomical_id(self, ticker: str) -> bytes | None:
        bp = self.session_mgr.bp
        height = bp.height
        ticker = ticker.lower()  # tickers are case-insensitive
        status, candidate_atomical_id, _ = bp.get_effective_ticker(ticker, height)
        if status == "verified":
            return candidate_atomical_id
        else:
            return None

    # check if this is atomicalId or ticker
    def _parse_atomical_id(self, id) -> bytes | None:
        if not id:
            return None
        atomical_id = None
        try:
            atomical_id = compact_to_location_id_bytes(id)
        except Exception:
            atomical_id = self._resolve_ticker_to_atomical_id(id)
            if not atomical_id:
                return None
        return atomical_id

    # check if this is address or pk_script
    def _parse_addr(self, wallet: str) -> str | None:
        if not wallet:
            return None
        addr = None
        try:
            pk_bytes = bytes.fromhex(wallet)
        except Exception:
            pk_bytes = None
        if pk_bytes:
            addr_from_pk = get_address_from_output_script(pk_bytes)
            if addr_from_pk:
                addr = addr_from_pk
        else:
            addr = wallet
        return addr

    # check if block height is valid
    def _parse_integer_string(self, input: str | int | None) -> int | None:
        # empty string
        if not input:
            return None
        if isinstance(input, int):
            return input
        # some character is not 0-9
        if not input.isnumeric():
            return None
        value = int(input)
        return value

    def _parse_limit_offset(
        self, limit_input: str | int | None, offset_input: str | int | None, default_limit: int, max_limit: int
    ) -> Tuple[int, int]:
        limit = self._parse_integer_string(limit_input)
        offset = self._parse_integer_string(offset_input)
        if limit is None:
            limit = default_limit
        if offset is None:
            offset = 0
        if limit > max_limit:
            raise BadRequestException(f"'limit' cannot exceed {max_limit}")
        return limit, offset

    # check if block range is valid (-1 means "latest block")
    def _parse_block_height_for_range(self, block_height_str: str) -> int | None:
        if block_height_str == "-1":
            return -1
        return self._parse_integer_string(block_height_str)

    def _block_height_to_unix_timestamp(self, block_height: int) -> int:
        db_block_ts = self.session_mgr.db.get_block_timestamp(block_height)
        if db_block_ts:
            return db_block_ts
        # TODO: should fall back to query timestamp from rpc if not found in db
        return 0

    @error_handler
    async def get_block_height(self, request: "Request") -> "Response":
        block_height = self.session_mgr.db.db_height
        (block_hash,) = await self.session_mgr.db.fs_block_hashes(block_height, 1)
        block_hash_hex = hash_to_hex_str(block_hash)
        return format_response(
            {
                "hash": block_hash_hex,
                "height": block_height,
            }
        )

    def _process_balance(self, address: str, balances: "dict[bytes, int]", tx_data):
        inputs: dict = tx_data["transfers"]["inputs"]
        outputs: dict = tx_data["transfers"]["outputs"]
        for _, input_atomicals in inputs.items():
            for input_atomical in input_atomicals:
                if input_atomical["address"] == address and input_atomical["type"] == "FT":
                    atomical_id_str = input_atomical["atomical_id"]
                    atomical_id = compact_to_location_id_bytes(atomical_id_str)
                    if atomical_id not in balances:
                        balances[atomical_id] = 0
                    balances[atomical_id] -= input_atomical["value"]
        for _, output_atomicals in outputs.items():
            for output_atomical in output_atomicals:
                if output_atomical["address"] == address and output_atomical["type"] == "FT":
                    atomical_id_str = output_atomical["atomical_id"]
                    atomical_id = compact_to_location_id_bytes(atomical_id_str)
                    if atomical_id not in balances:
                        balances[atomical_id] = 0
                    balances[atomical_id] += output_atomical["value"]
        if tx_data["op"] in ["mint-dft", "mint-ft"]:
            # minted fts is always at output index 0
            minted_fts = tx_data["info"]["outputs"][0]
            for minted_ft in minted_fts:
                atomical_id_str = minted_ft["atomical_id"]
                atomical_id = compact_to_location_id_bytes(atomical_id_str)
                if atomical_id not in balances:
                    balances[atomical_id] = 0
                balances[atomical_id] += minted_ft["value"]

    async def _get_atomical(self, atomical_id: bytes) -> "dict":
        compact_atomical_id = location_id_bytes_to_compact(atomical_id)
        atomical = await self.session_mgr.atomical_id_get(compact_atomical_id)
        return atomical

    async def _confirmed_history(self, hashX):
        # Note history is ordered
        history, _ = await self.session_mgr.limited_history(hashX)
        conf = [{"tx_hash": hash_to_hex_str(tx_hash), "height": height} for tx_hash, height in history]
        return conf

    async def _get_populated_arc20_balances(
        self, address: str, atomical_id: "bytes | None", block_height: int, limit: int = None, offset: int = 0
    ):
        pk_scriptb = get_script_from_address(address)

        balances: "dict[bytes, int]" = {}  # atomical_id -> amount (int)
        script_hash = sha256(pk_scriptb)
        hashX = scripthash_to_hashX(script_hash)
        if not hashX:
            raise Exception("Invalid hashX")  # should not happen since we are using sha256
        history_data = await self._confirmed_history(hashX)
        # only use transactions after ATOMICALS_ACTIVATION_HEIGHT and before block_height
        history_data = [
            x
            for x in history_data
            if self.session_mgr.env.coin.ATOMICALS_ACTIVATION_HEIGHT <= x["height"] and x["height"] <= block_height
        ]

        history_list = []
        for history in list(history_data):
            tx_num, _ = self.session_mgr.db.get_tx_num_height_from_tx_hash(hex_str_to_hash(history["tx_hash"]))
            history["tx_num"] = tx_num
            history_list.append(history)

        history_list.sort(key=lambda x: x["tx_num"])
        tx_datas = await asyncio.gather(
            *[
                self.session_mgr.get_transaction_detail(history["tx_hash"], history["height"], history["tx_num"])
                for history in history_list
            ]
        )
        for tx_data in tx_datas:
            self._process_balance(address, balances, tx_data)

        # clear empty balances and filter by atomical_id
        balances = {k: v for k, v in balances.items() if v != 0 and (not atomical_id or k == atomical_id)}

        balances_list = list(balances.items())
        balances_list.sort(key=lambda x: (x[1], x[0]), reverse=True)  # sort by amount and atomical_id desc
        if limit is not None:
            balances_list = balances_list[offset : offset + limit]
        # populate atomical objects
        atomical_ids = [atomical_id for atomical_id, _ in balances_list]
        atomicals_list = await asyncio.gather(*[self._get_atomical(atomical_id) for atomical_id in atomical_ids])
        atomicals: dict[bytes, dict] = {
            atomical_id: atomical for atomical_id, atomical in zip(atomical_ids, atomicals_list, strict=True)
        }

        populated_balances: "list[dict]" = []  # atomical_id -> { "amount": int, "ticker": str | None }
        for atomical_id, amount in balances_list:
            atomical = atomicals.get(atomical_id)
            if atomical:
                ticker = atomical.get("$ticker", "")
                balance = {
                    "amount": str(amount),
                    "id": location_id_bytes_to_compact(atomical_id),
                    "name": ticker,
                    "symbol": ticker,
                    "decimals": get_decimals(),
                }
                populated_balances.append(balance)
        return populated_balances

    @error_handler
    async def get_arc20_balance(self, request: "Request") -> "Response":
        # parse wallet
        wallet = request.match_info.get("wallet", "")
        if not wallet:
            return format_response(None, 400, "Wallet is required.")
        address = self._parse_addr(wallet)
        if not address:
            return format_response(None, 400, "Invalid wallet.")

        # parse block_height
        latest_block_height = self.session_mgr.db.db_height
        q_block_height = request.query.get("blockHeight")
        block_height = latest_block_height
        if q_block_height is not None:
            block_height = self._parse_integer_string(q_block_height)
            if block_height is None:
                return format_response(None, 400, "Invalid block height.")

        # parse atomical_id
        id = request.query.get("id")
        atomical_id = None
        if id:
            atomical_id = self._parse_atomical_id(id)
            if not atomical_id:
                return format_response(None, 400, "Invalid ID.")

        limit, offset = self._parse_limit_offset(
            request.query.get("limit"),
            request.query.get("offset"),
            DEFAULT_LIMIT.get_arc20_balances,
            MAX_LIMIT.get_arc20_balances,
        )

        populated_balances = await self._get_populated_arc20_balances(address, atomical_id, block_height, limit, offset)
        return format_response({"blockHeight": block_height, "list": populated_balances})

    @error_handler
    async def get_arc20_balances_batch(self, request: "Request") -> "Response":
        body: dict = await request.json()
        queries: "list[BalanceQuery]" = []
        latest_block_height = self.session_mgr.db.db_height

        raw_queries: "list[dict]" = body.get("queries", [])
        if len(raw_queries) > MAX_LIMIT.get_arc20_balances_batch_queries:
            return format_response(None, 400, f"cannot exceed {MAX_LIMIT.get_arc20_balances_batch_queries} queries.")
        for idx, raw_query in enumerate(raw_queries):
            # parse wallet
            wallet = raw_query.get("wallet", "")
            if not wallet:
                return format_response(None, 400, f"query index {idx}: wallet is required.")
            address = self._parse_addr(wallet)
            if not address:
                return format_response(None, 400, f"query index {idx}: invalid wallet.")

            # parse block_height
            # this is from json, so q_block_height may be number itself
            q_block_height = raw_query.get("blockHeight")
            block_height = latest_block_height
            if q_block_height is not None:
                block_height = self._parse_integer_string(str(q_block_height))
                if block_height is None:
                    return format_response(None, 400, f"query index {idx}: invalid block height.")

            # parse atomical_id
            id = raw_query.get("id")
            atomical_id = None
            if id:
                atomical_id = self._parse_atomical_id(id)
                if not atomical_id:
                    return format_response(None, 400, f"query index {idx}: invalid ID.")

            limit, offset = self._parse_limit_offset(
                raw_query.get("limit"),
                raw_query.get("offset"),
                DEFAULT_LIMIT.get_arc20_balances,
                MAX_LIMIT.get_arc20_balances,
            )

            # append to list
            queries.append(BalanceQuery(address, atomical_id, block_height, limit, offset))

        results = await asyncio.gather(
            *[
                self._get_populated_arc20_balances(
                    query.address, query.atomical_id, query.block_height, query.limit, query.offset
                )
                for query in queries
            ]
        )
        formatted_results = [
            {"blockHeight": query.block_height, "list": result} for query, result in zip(queries, results, strict=True)
        ]
        return format_response({"list": formatted_results})

    async def _get_tx_detail(self, tx_hash: str, f_atomical_id: bytes | None, f_address: str | None) -> dict | None:
        tx_data = await self.session_mgr.get_transaction_detail(tx_hash)
        block_height = tx_data.get("height", 0)
        tx_info: dict = tx_data.get("info", {})
        tx_transfers: dict = tx_data.get("transfers", {})
        tx_op = tx_data.get("op", "")

        # filter only supported op
        if tx_op not in supported_ops:
            return None

        tx_payload: dict = tx_info.get("payload", {})
        tx_payload_args: dict = tx_payload.get("args", {})

        # tx_transfers distribution
        # contains inputs, outputs, is_burned, burned_fts, is_cleanly_assigned
        tx_inputs: dict[int, list[dict]] = tx_transfers.get("inputs", {})
        tx_outputs: dict[int, list[dict]] = tx_transfers.get("outputs", {})
        tx_burned_fts: dict = tx_transfers.get("burned_fts", {})

        commit_tx_id = None
        commit_index = None

        tx_hashb = hex_str_to_hash(tx_hash)
        raw_tx = self.session_mgr.db.get_raw_tx_by_tx_hash(tx_hashb)
        tx, _tx_hash = self.session_mgr.env.coin.DESERIALIZER(raw_tx, 0).read_tx_and_hash()
        assert tx_hashb == _tx_hash
        operation_found_at_inputs = parse_protocols_operations_from_witness_array(tx, tx_hash, True)
        if operation_found_at_inputs:
            commit_tx_id = hash_to_hex_str(operation_found_at_inputs.get("commit_txid"))
            commit_index = operation_found_at_inputs.get("commit_index")

        inputs = []
        # if multiple FTs in same utxo,
        # will have multiple inputs with same index
        for tx_i in tx_inputs.values():
            for tx_input in tx_i:
                address = tx_input.get("address", "")
                pk_script = ""
                if address:
                    pk_script = get_script_from_address(address).hex()
                input_map = {
                    "index": tx_input.get("index", 0),
                    "id": tx_input.get("atomical_id", ""),
                    "amount": str(tx_input.get("value", 0)),
                    "decimals": get_decimals(),
                    "address": address,
                    "pkScript": pk_script,
                }
                inputs.append(input_map)

        outputs = []
        # includes FT outputs from mints too
        for tx_o in tx_outputs.values():
            for tx_output in tx_o:
                address = tx_output.get("address", "")
                pk_script = ""
                if address:
                    pk_script = get_script_from_address(address).hex()
                output_map = {
                    "index": tx_output.get("index", 0),
                    "id": tx_output.get("atomical_id", ""),
                    "amount": str(tx_output.get("value", 0)),
                    "decimals": get_decimals(),
                    "address": address,
                    "pkScript": pk_script,
                }
                outputs.append(output_map)

        mints = {}
        mint_ticker: str = tx_payload_args.get("mint_ticker", "")
        if mint_ticker:
            tx_info_outputs: dict = tx_info.get("outputs", {})
            mint_outputs: list[dict] = tx_info_outputs.get(0, [])  # mint output should be in index 0
            for output_data in mint_outputs:
                atomical_id_str = output_data.get("atomical_id", "")
                address = output_data.get("address", "")
                mint_amount = output_data.get("value", 0)
                pk_script = ""
                if address:
                    pk_script = get_script_from_address(address).hex()
                output_map = {
                    "index": output_data.get("index", ""),
                    "id": atomical_id_str,
                    "amount": str(mint_amount),
                    "decimals": get_decimals(),
                    "address": address,
                    "pkScript": pk_script,
                }
                outputs.append(output_map)
                prev_mint: dict | None = mints.get(atomical_id_str, None)
                if not prev_mint:
                    prev_mint = {
                        "amount": "0",
                        "decimals": get_decimals(),
                    }
                new_mint = {
                    "amount": str(int(prev_mint["amount"]) + mint_amount),
                    "decimals": get_decimals(),
                }
                mints[atomical_id_str] = new_mint

        burns = {}
        for k, v in tx_burned_fts.items():
            burns[k] = {
                "amount": str(v),
                "decimals": get_decimals(),
            }

        # sort inputs and outputs asc
        inputs.sort(key=lambda x: x["index"], reverse=False)
        outputs.sort(key=lambda x: x["index"], reverse=False)

        # filter checking
        if f_address:
            found_input = f_address in [e["address"] for e in inputs]
            found_output = f_address in [e["address"] for e in outputs]
            if not (found_input or found_output):
                return None

        if f_atomical_id:
            f_atomical_id_str = location_id_bytes_to_compact(f_atomical_id)
            # only dft, must check atomical related field `commit_tx_id`
            if tx_op == "dft":
                this_atomical_id = commit_tx_id + "i" + str(commit_index)
                if this_atomical_id != f_atomical_id_str:
                    return None
            else:
                found_input = f_atomical_id_str in [e["id"] for e in inputs]
                fount_output = f_atomical_id_str in [e["id"] for e in outputs]
                if not (found_input or fount_output):
                    return None

        tx_index_in_block = self.session_mgr.db.get_tx_index_from_tx_hash(hex_str_to_hash(tx_hash))
        if not tx_index_in_block:
            tx_index_in_block = 0

        return {
            "txHash": tx_hash,
            "blockHeight": block_height,
            "index": tx_index_in_block,
            "timestamp": self._block_height_to_unix_timestamp(block_height),
            "inputs": inputs,
            "outputs": outputs,
            "mints": mints,
            "burns": burns,
            # arc20-specific data
            "extend": {
                "op": tx_op,
                "info": {
                    "payload": tx_payload,
                },
                "commitTxHash": commit_tx_id,
                "commitIndex": commit_index,
            },
        }

    async def get_txs_from_history_limited(
        self,
        hashX,
        from_block: int,
        to_block: int,
        f_atomical_id: bytes | None,
        f_address: str | None,
        limit=None,
        reverse=True,  # set to true for descending order
    ) -> list:
        txs = []
        txnum_padding = bytes(8 - TXNUM_LEN)
        for _, hist in self.session_mgr.db.history.db.iterator(prefix=hashX, reverse=reverse):
            for tx_numb in util.chunks(hist, TXNUM_LEN):
                (tx_num,) = util.unpack_le_uint64(tx_numb + txnum_padding)
                tx_hash, height = self.session_mgr.db.fs_tx_hash(tx_num)
                # skip txs outside block range
                if not (from_block <= height <= to_block):
                    continue

                # check tx has atomical operation, so we don't waste time on daemon for non-atomical txs
                op_data = self.session_mgr._tx_num_op_cache.get(tx_num)
                if not op_data:
                    op_prefix_key = b"op" + util.pack_le_uint64(tx_num)
                    tx_op = self.session_mgr.db.utxo_db.get(op_prefix_key)
                    if tx_op:
                        (op_data,) = util.unpack_le_uint32(tx_op)
                        self.session_mgr._tx_num_op_cache[tx_num] = op_data

                # append only txs with atomical operation
                if not op_data:
                    continue
                tx_hash_str = hash_to_hex_str(tx_hash)
                tx = await self._get_tx_detail(tx_hash_str, f_atomical_id, f_address)
                if tx:
                    txs.append(tx)

            # only break when all tx_nums in hist are processed, since tx_nums are always sorted in ascending order and we need all of them if reverse=True
            if limit is not None and len(txs) >= limit:
                break
        if reverse:
            txs.sort(key=lambda x: (x["blockHeight"], x["index"]), reverse=reverse)
        if limit is not None:
            txs = txs[:limit]
        return txs

    @error_handler
    async def get_arc20_transactions(self, request: "Request") -> "Response":
        # parse wallet (optional)
        wallet = request.query.get("wallet")
        address = None
        if wallet is not None:
            address = self._parse_addr(wallet)
            if not address:
                return format_response(None, 400, "Invalid wallet.")

        # parse block heights filter
        q_from_block = request.query.get("fromBlock")
        q_to_block = request.query.get("toBlock")
        from_block = 0
        to_block = -1
        if q_from_block is not None:
            from_block = self._parse_block_height_for_range(q_from_block)
            if from_block is None:
                return format_response(None, 400, "Invalid fromBlock.")
        if q_to_block is not None:
            to_block = self._parse_block_height_for_range(q_to_block)
            if to_block is None:
                return format_response(None, 400, "Invalid toBlock.")

        # parse atomical_id
        id = request.query.get("id")
        atomical_id = None
        if id:
            atomical_id = self._parse_atomical_id(id)
            if not atomical_id:
                return format_response(None, 400, "Invalid ID.")

        limit, offset = self._parse_limit_offset(
            request.query.get("limit"),
            request.query.get("offset"),
            DEFAULT_LIMIT.get_arc20_transactions,
            MAX_LIMIT.get_arc20_transactions,
        )

        latest_block_height = self.session_mgr.db.db_height
        if from_block == -1:
            from_block = latest_block_height
        if to_block == -1:
            to_block = latest_block_height
        if from_block > to_block:
            return format_response(None, 400, "fromBlock must be <= toBlock.")

        txs = []

        # TODO: remove this limit when we can fix this performance issue
        # temporary limit for large queries
        max_limit_offset = 5000
        if limit + offset > max_limit_offset:
            return format_response(None, 400, f"limit + offset must be <= {max_limit_offset}.")

        # if queries for single block, use that first
        if from_block == to_block:
            block_txs = self.session_mgr.db.get_atomicals_block_txs_with_tx_num(from_block)
            block_txs.sort(key=lambda x: x["tx_num"], reverse=True)
            for block_tx in block_txs:
                tx_hash = block_tx["tx_hash"]
                tx = await self._get_tx_detail(tx_hash, atomical_id, address)
                if tx:
                    txs.append(tx)
                    if len(txs) >= limit + offset:
                        break

        elif address:
            hashX = scripthash_to_hashX(sha256(get_script_from_address(address)))
            # use address = None to skip filtering check
            txs = await self.get_txs_from_history_limited(
                hashX, from_block, to_block, atomical_id, None, limit + offset, reverse=True
            )  # get latest txs

        elif atomical_id:
            # get all tx filter by id
            hashX = double_sha256(atomical_id)
            # use atomical_id = None to skip filtering check
            txs = await self.get_txs_from_history_limited(
                hashX, from_block, to_block, None, address, limit + offset, reverse=True
            )  # get latest txs

        # query block range sequentially, starting from to_block, until limit is reached
        else:
            for block_height in range(to_block, from_block - 1, -1):
                block_txs = self.session_mgr.db.get_atomicals_block_txs_with_tx_num(block_height)
                block_txs.sort(key=lambda x: x["tx_num"], reverse=True)
                for block_tx in block_txs:
                    tx_hash = block_tx["tx_hash"]
                    tx = await self._get_tx_detail(tx_hash, atomical_id, address)
                    if tx:
                        txs.append(tx)
                        if len(txs) >= limit + offset:
                            break
                if len(txs) >= limit + offset:
                    break

        # assumes txs is ALREADY SORTED in descending order!
        txs = txs[offset : offset + limit]

        return format_response(
            {
                "list": txs,
            }
        )

    async def _get_arc20_holders_by_block_height(self, atomical_id: bytes, block_height: int) -> dict:
        utxos = await self.session_mgr.db.get_atomical_utxos_at_height_by_atomical_id(atomical_id, block_height)

        total_value = 0
        holder_map: dict[bytes, int] = {}

        # group by pk_script and map to address
        for utxo in utxos:
            prev_value = holder_map.get(utxo.pk_script, 0)
            holder_map[utxo.pk_script] = prev_value + utxo.atomical_value
            total_value = total_value + utxo.atomical_value

        return {
            "total": total_value,
            "count": len(holder_map),
            "holders": holder_map,
        }

    @error_handler
    async def get_arc20_holders(self, request: "Request") -> "Response":
        # parse atomical_id
        id = request.match_info.get("id", "")
        if not id:
            return format_response(None, 400, "ID is required.")
        atomical_id = self._parse_atomical_id(id)
        if not atomical_id:
            return format_response(None, 400, "Invalid ID.")

        # parse block_height
        latest_block_height = self.session_mgr.db.db_height
        q_block_height = request.query.get("blockHeight")
        block_height = latest_block_height
        if q_block_height is not None:
            block_height = self._parse_integer_string(q_block_height)
            if block_height is None:
                return format_response(None, 400, "Invalid block height.")

        limit, offset = self._parse_limit_offset(
            request.query.get("limit"),
            request.query.get("offset"),
            DEFAULT_LIMIT.get_arc20_holders,
            MAX_LIMIT.get_arc20_holders,
        )

        # base data
        atomical = await self._get_atomical(atomical_id)

        atomical_type = atomical.get("type", "")
        if atomical_type == "NFT":
            return format_response(None, 400, "NFT is not supported.")

        subtype = atomical.get("subtype", "")
        mint_mode = atomical.get("$mint_mode", "")
        mint_info: dict = atomical.get("mint_info", {})
        mint_info_args: dict = mint_info.get("args", {})

        max_supply = 0
        mint_amount = 0
        minted_amount = 0

        # find max_supply and minted_amount
        if atomical_type == "FT":
            if mint_mode == "fixed":
                max_supply = atomical.get("$max_supply", 0)
                mint_amount = mint_info_args.get("mint_amount", 0)
            else:
                max_supply = atomical.get("$max_supply", -1)
                if max_supply < 0:
                    mint_amount = mint_info_args.get("mint_amount", 0)
                    max_supply = DFT_MINT_MAX_MAX_COUNT_DENSITY * mint_amount
        # NOTE: unsupported
        # elif atomical_type == "NFT":
        else:
            raise Exception("unreachable code: invalid atomical type")

        if subtype == "decentralized":
            mint_count = self.session_mgr.bp.get_distmints_count_by_atomical_id(block_height, atomical_id, True)
            minted_amount = mint_count * mint_amount  # total minted
        elif subtype == "direct":
            minted_amount = max_supply  # entire mint in direct mint
        else:
            raise Exception("unreachable code: invalid subtype")

        holders: list[Tuple[bytes, int]] = []
        # support only atomical FT
        if atomical_type == "FT":
            if block_height == latest_block_height:
                atomical: dict = await self.session_mgr.db.populate_extended_atomical_holder_info(atomical_id, atomical)
                holders = [
                    (bytes.fromhex(holder["script"]), holder.get("holding", 0))
                    for holder in atomical.get("holders", [])
                ]
            else:
                # get historical data
                data = await self._get_arc20_holders_by_block_height(atomical_id, block_height)
                holders = [(pk_script, amount) for pk_script, amount in data["holders"].items()]

        # sort by holding desc
        holders.sort(key=lambda x: (x[1], x[0]), reverse=True)
        holders = holders[offset : offset + limit]

        formatted_results = []
        for pk_scriptb, amount in holders:
            address = get_address_from_output_script(pk_scriptb)
            formatted_results.append(
                {
                    "address": address if address else "",
                    "pkScript": pk_scriptb.hex(),
                    "amount": str(amount),
                    "percent": amount / max_supply,
                }
            )

        return format_response(
            {
                "blockHeight": block_height,
                "totalSupply": str(max_supply),
                "mintedAmount": str(minted_amount),
                "decimals": get_decimals(),
                "list": formatted_results,
            }
        )

    @error_handler
    async def get_arc20_token(self, request: "Request") -> "Response":
        # parse atomical_id
        id = request.match_info.get("id", "")
        if not id:
            return format_response(None, 400, "ID is required.")
        atomical_id = self._parse_atomical_id(id)
        if not atomical_id:
            return format_response(None, 400, "Invalid ID.")

        # parse block_height
        latest_block_height = self.session_mgr.db.db_height
        q_block_height = request.query.get("blockHeight")
        block_height = latest_block_height
        if q_block_height is not None:
            block_height = self._parse_integer_string(q_block_height)
            if block_height is None:
                return format_response(None, 400, "Invalid block height.")

        # get data
        atomical: dict = await self._get_atomical(atomical_id)

        atomical_type = atomical.get("type", "")
        if atomical_type == "NFT":
            return format_response(None, 400, "NFT is not supported.")

        subtype = atomical.get("subtype", "")
        mint_mode = atomical.get("$mint_mode", "")
        mint_info: dict = atomical.get("mint_info", {})
        mint_info_args: dict = mint_info.get("args", {})
        ticker = atomical.get("$ticker", "")

        mint_count = 0
        minted_amount = 0
        max_supply = 0
        mint_amount = 0  # mint size

        if atomical_type == "FT":
            if mint_mode == "fixed":
                max_supply = atomical.get("$max_supply", 0)
                mint_amount = mint_info_args.get("mint_amount", 0)
            else:
                max_supply = atomical.get("$max_supply", -1)
        # NOTE: unsupported
        # elif atomical_type == "NFT":
        else:
            raise Exception("unreachable code: invalid atomical type")

        if subtype == "decentralized":
            atomical: dict = await self.session_mgr.bp.get_dft_mint_info_rpc_format_by_atomical_id(atomical_id)
            mint_count = atomical["dft_info"]["mint_count"]
            minted_amount = mint_count * mint_amount  # total minted
        elif subtype == "direct":
            atomical: dict = await self.session_mgr.bp.get_ft_mint_info_rpc_format_by_atomical_id(atomical_id)
            minted_amount = max_supply  # entire mint in direct mint
        else:
            raise Exception("unreachable code: invalid subtype")

        location_summary: dict = atomical.get("location_summary", {})
        holder_count = location_summary.get("unique_holders", 0)
        circulating_supply = location_summary.get("circulating_supply", 0)

        # deployment data
        commit_tx_id = mint_info.get("commit_txid")
        commit_tx_height = mint_info.get("commit_height")

        reveal_location_script: str = mint_info.get("reveal_location_script")
        deployer_address = get_address_from_output_script(bytes.fromhex(reveal_location_script))
        if not deployer_address:
            deployer_address = ""

        deployed_at_height = commit_tx_height
        deployed_at = self._block_height_to_unix_timestamp(deployed_at_height)
        deploy_tx_hash = commit_tx_id

        # change time-sensitive data from block_height
        if block_height != latest_block_height:
            # support only atomical FT
            data = await self._get_arc20_holders_by_block_height(atomical_id, block_height)
            holder_count = data["count"]
            circulating_supply = data["total"]
            if subtype == "decentralized":
                mint_count = await self.session_mgr.db.get_atomical_mint_count_at_height(atomical_id, block_height)
                minted_amount = mint_count * mint_amount

        # mint completion data
        completed_at_height = None
        completed_at = None  # unix timestamp
        if minted_amount == max_supply:
            if subtype == "direct":
                completed_at_height = deployed_at_height
                completed_at = deployed_at
            else:
                completed_at_height = self.session_mgr.db.get_mint_completed_at_height(atomical_id)
                if completed_at_height:
                    completed_at = self._block_height_to_unix_timestamp(completed_at_height)
        # clear historical data
        if completed_at_height and completed_at_height > block_height:
            completed_at_height = None
            completed_at = None

        compact_atomical_id = location_id_bytes_to_compact(atomical_id)
        return format_response(
            {
                "id": compact_atomical_id,
                "name": ticker,
                "symbol": ticker,
                "totalSupply": str(max_supply),
                "circulatingSupply": str(circulating_supply),
                "mintedAmount": str(minted_amount),
                "burnedAmount": str(minted_amount - circulating_supply),
                "decimals": get_decimals(),
                "deployedAt": deployed_at,
                "deployedAtHeight": deployed_at_height,
                "deployTxHash": deploy_tx_hash,
                "completedAt": completed_at,
                "completedAtHeight": completed_at_height,
                "holdersCount": holder_count,
                # arc20-specific data
                "extend": {
                    "atomicalId": compact_atomical_id,
                    "atomicalNumber": atomical.get("atomical_number", 0),
                    "atomicalRef": atomical.get("atomical_ref", ""),
                    "amountPerMint": str(mint_amount),
                    "maxMints": str(atomical.get("$max_mints", 0)),  # number of times this token can be minted
                    "deployedBy": deployer_address,
                    "mintHeight": atomical.get("$mint_height", 0),  # the block height this FT can start to be minted
                    "mintInfo": {
                        "commitTxHash": commit_tx_id,
                        # commit tx output index of utxo used in reveal tx
                        "commitIndex": mint_info.get("commit_index"),
                        "revealTxHash": mint_info.get("reveal_location_txid"),
                        "revealIndex": mint_info.get("reveal_location_index"),
                        "args": mint_info_args,  # raw atomicals operation payload
                        "metadata": mint_info.get("meta", {}),  # metadata.json used during deployment
                    },
                    "subtype": subtype,
                    "mintMode": mint_mode,
                },
            }
        )

    @error_handler
    async def get_arc20_token_list(self, request: "Request") -> "Response":
        limit, offset = self._parse_limit_offset(
            request.query.get("limit"),
            request.query.get("offset"),
            DEFAULT_LIMIT.get_arc20_token_list,
            MAX_LIMIT.get_arc20_token_list,
        )


        infos = []
        atomical_ids = await self._get_atomicals_ft_list(limit,offset,asc=True)
        block_height = self.session_mgr.db.db_height
        for atomical_id in atomical_ids:
            # get data
            atomical: dict = await self._get_atomical(atomical_id)

            atomical_type = atomical.get("type", "")
            if atomical_type == "NFT":
                continue

            subtype = atomical.get("subtype", "")
            mint_mode = atomical.get("$mint_mode", "")
            mint_info: dict = atomical.get("mint_info", {})
            mint_info_args: dict = mint_info.get("args", {})
            ticker = atomical.get("$ticker", "")

            mint_count = 0
            minted_amount = 0
            max_supply = 0
            mint_amount = 0  # mint size

            if atomical_type == "FT":
                if mint_mode == "fixed":
                    max_supply = atomical.get("$max_supply", 0)
                    mint_amount = mint_info_args.get("mint_amount", 0)
                else:
                    max_supply = atomical.get("$max_supply", -1)
            # NOTE: unsupported
            # elif atomical_type == "NFT":
            else:
                raise Exception("unreachable code: invalid atomical type")

            if subtype == "decentralized":
                atomical: dict = await self.session_mgr.bp.get_dft_mint_info_rpc_format_by_atomical_id(atomical_id)
                mint_count = atomical["dft_info"]["mint_count"]
                minted_amount = mint_count * mint_amount  # total minted
            elif subtype == "direct":
                atomical: dict = await self.session_mgr.bp.get_ft_mint_info_rpc_format_by_atomical_id(atomical_id)
                minted_amount = max_supply  # entire mint in direct mint
            else:
                raise Exception("unreachable code: invalid subtype")

            location_summary: dict = atomical.get("location_summary", {})
            holder_count = location_summary.get("unique_holders", 0)
            circulating_supply = location_summary.get("circulating_supply", 0)

            # deployment data
            commit_tx_id = mint_info.get("commit_txid")
            commit_tx_height = mint_info.get("commit_height")

            reveal_location_script: str = mint_info.get("reveal_location_script")
            deployer_address = get_address_from_output_script(bytes.fromhex(reveal_location_script))
            if not deployer_address:
                deployer_address = ""

            deployed_at_height = commit_tx_height
            deployed_at = self._block_height_to_unix_timestamp(deployed_at_height)
            deploy_tx_hash = commit_tx_id

            # mint completion data
            completed_at_height = None
            completed_at = None  # unix timestamp
            if minted_amount == max_supply:
                if subtype == "direct":
                    completed_at_height = deployed_at_height
                    completed_at = deployed_at
                else:
                    completed_at_height = self.session_mgr.db.get_mint_completed_at_height(atomical_id)
                    if completed_at_height:
                        completed_at = self._block_height_to_unix_timestamp(completed_at_height)
            # clear historical data
            if completed_at_height and completed_at_height > block_height:
                completed_at_height = None
                completed_at = None

            compact_atomical_id = location_id_bytes_to_compact(atomical_id)

            infos.append({
                "id": compact_atomical_id,
                "name": ticker,
                "symbol": ticker,
                "totalSupply": str(max_supply),
                "circulatingSupply": str(circulating_supply),
                "mintedAmount": str(minted_amount),
                "burnedAmount": str(minted_amount - circulating_supply),
                "decimals": get_decimals(),
                "deployedAt": deployed_at,
                "deployedAtHeight": deployed_at_height,
                "deployTxHash": deploy_tx_hash,
                "completedAt": completed_at,
                "completedAtHeight": completed_at_height,
                "holdersCount": holder_count,
                # arc20-specific data
                "extend": {
                    "atomicalId": compact_atomical_id,
                    "atomicalNumber": atomical.get("atomical_number", 0),
                    "atomicalRef": atomical.get("atomical_ref", ""),
                    "amountPerMint": str(mint_amount),
                    "maxMints": str(atomical.get("$max_mints", 0)),  # number of times this token can be minted
                    "deployedBy": deployer_address,
                    "mintHeight": atomical.get("$mint_height", 0),  # the block height this FT can start to be minted
                    "mintInfo": {
                        "commitTxHash": commit_tx_id,
                        # commit tx output index of utxo used in reveal tx
                        "commitIndex": mint_info.get("commit_index"),
                        "revealTxHash": mint_info.get("reveal_location_txid"),
                        "revealIndex": mint_info.get("reveal_location_index"),
                        "args": mint_info_args,  # raw atomicals operation payload
                        "metadata": mint_info.get("meta", {}),  # metadata.json used during deployment
                    },
                    "subtype": subtype,
                    "mintMode": mint_mode,
                },
            })
        return format_response({
            "list": infos,
        })
    async def _get_atomicals_ft_list(self, limit, offset, asc=True) -> list:
        atomical_ids = await self.session_mgr.db.get_atomicals_list(limit, offset, asc)
        # TODO:
        # - implement logic from `electrumx/server/db.py.get_atomicals_list()`
        # - each iteration should call `self.session_mgr.bp.get_atomicals_id_mint_info(atomical_id, True) to check type is FT or NFT`
        return atomical_ids

    async def _utxo_to_formatted(self, utxo: UTXO) -> "dict":
        # TODO: use data from AtomicalUTXO only when it has atomical_id in it
        tx_id_str = hash_to_hex_str(utxo.tx_hash)
        output_index = utxo.tx_pos
        location = compact_to_location_id_bytes(tx_id_str + "i" + str(output_index))
        atomicals_found_at_location = self.session_mgr.db.get_atomicals_by_location_extended_info_long_form(location)
        atomical_ids: list = atomicals_found_at_location.get("atomicals", [])
        atomical_values: dict[bytes, int] = {}
        for atomical_id in atomicals_found_at_location["atomicals"]:
            atomical_values[atomical_id] = self.session_mgr.db.get_uxto_atomicals_value(location, atomical_id)

        formatted_atomicals = []
        # should be 0 or 1 item
        for atomical_id in atomical_ids:
            atomical = await self._get_atomical(atomical_id)
            ticker = atomical.get("$ticker", "")
            atomical_type = atomical.get("type", "")
            atomical_out = {
                "atomicalId": location_id_bytes_to_compact(atomical_id),
                "type": atomical_type,
            }
            if atomical_type == "FT":
                atomical_out["ftTicker"] = ticker
                atomical_out["ftAmount"] = str(atomical_values.get(atomical_id, 0))
                atomical_out["ftDecimals"] = get_decimals()
            formatted_atomicals.append(atomical_out)
        res = {
            "txHash": tx_id_str,
            "outputIndex": utxo.tx_pos,
            "sats": utxo.value,
            "extend": {"atomicals": formatted_atomicals},
        }
        return res

    @error_handler
    async def get_arc20_utxos(self, request: "Request") -> "Response":
        # parse wallet
        wallet = request.match_info.get("wallet", "")
        if not wallet:
            return format_response(None, 400, "Wallet is required.")
        address = self._parse_addr(wallet)
        if not address:
            return format_response(None, 400, "Invalid wallet.")
        pk_scriptb = get_script_from_address(address)

        # parse block_height
        latest_block_height = self.session_mgr.db.db_height
        q_block_height = request.query.get("blockHeight")
        block_height = latest_block_height
        if q_block_height is not None:
            block_height = self._parse_integer_string(q_block_height)
            if block_height is None:
                return format_response(None, 400, "Invalid block height.")

        # parse atomical_id
        id = request.query.get("id")
        atomical_id = None
        if id:
            atomical_id = self._parse_atomical_id(id)
            if not atomical_id:
                return format_response(None, 400, "Invalid ID.")

        limit, offset = self._parse_limit_offset(
            request.query.get("limit"),
            request.query.get("offset"),
            DEFAULT_LIMIT.get_arc20_utxos,
            MAX_LIMIT.get_arc20_utxos,
        )

        formatted_results: list[dict] = []
        utxos: list[UTXO] | list[AtomicalUTXO] = []

        # bypass for latest_block (quicker)
        if block_height == latest_block_height:
            hashX = scripthash_to_hashX(sha256(pk_scriptb))
            utxos = await self.session_mgr.db.all_utxos(hashX)
        else:
            atomical_utxos = await self.session_mgr.db.get_atomical_utxos_at_height_by_pk_script(
                pk_scriptb, block_height
            )
            # cast AtomicalUTXO to UTXO
            # TODO: switch to using AtomicalUTXO in _utxo_to_formatted() instead
            seen_outpoints: set[Tuple[bytes, int]] = set()
            utxos = []
            for atomical_utxo in atomical_utxos:
                if (atomical_utxo.tx_hash, atomical_utxo.tx_pos) in seen_outpoints:
                    continue
                seen_outpoints.add((atomical_utxo.tx_hash, atomical_utxo.tx_pos))
                utxos.append(
                    UTXO(-1, atomical_utxo.tx_pos, atomical_utxo.tx_hash, atomical_utxo.height, atomical_utxo.sat_value)
                )

        formatted_results = await asyncio.gather(*[self._utxo_to_formatted(utxo) for utxo in utxos])

        # return only UTXOs that contain atomical, or filter if parameter passed
        filtered_formatted = []
        for e in formatted_results:
            atomical_list: list = e["extend"]["atomicals"]
            # skip UTXO that does not contain atomical
            if len(atomical_list) == 0:
                continue
            found = True
            if atomical_id:
                atomical_id_str = location_id_bytes_to_compact(atomical_id)
                found = atomical_id_str in [a["atomicalId"] for a in atomical_list]
            if found:
                filtered_formatted.append(e)

        filtered_formatted = filtered_formatted[offset : offset + limit]
        return format_response(
            {
                "blockHeight": block_height,
                "list": filtered_formatted,
            }
        )
