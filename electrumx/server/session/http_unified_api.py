import asyncio
import base64
from dataclasses import dataclass
import json
from typing import TYPE_CHECKING, Optional
from aiohttp.web import json_response
from aiohttp.web_urldispatcher import UrlDispatcher

from electrumx.lib import util
from electrumx.lib.hash import HASHX_LEN, double_sha256, hash_to_hex_str, hex_str_to_hash, sha256
from electrumx.lib.script2addr import get_address_from_output_script, get_script_from_address
from electrumx.lib.util_atomicals import DFT_MINT_MAX_MAX_COUNT_DENSITY, compact_to_location_id_bytes, location_id_bytes_to_compact, parse_protocols_operations_from_witness_array
from electrumx.server.db import UTXO

if TYPE_CHECKING:
    from electrumx.server.controller import SessionManager
    from aiohttp.web import Request, Response

class JSONBytesEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, bytes):
            # use base64 encoding for bytes
            return base64.b64encode(obj).decode()
        return super().default(obj)

def format_response(result: "dict | None", status: "int | None" = None, error: "str | None" = None) -> "Response":
    if error:
        if status == None:
            status = 500
        return json_response({
                "error": error,
            },
            status=status,
            dumps=lambda o: json.dumps(o, cls=JSONBytesEncoder)
        )
    if status == None:
        status = 200
    return json_response({
                "error": None,
                "result": result,
            },
            status=status,
            dumps=lambda o: json.dumps(o, cls=JSONBytesEncoder)
        )

def scripthash_to_hashX(script_hash: bytes) -> "Optional[bytes]":
    if len(script_hash) == 32:
        return script_hash[:HASHX_LEN]
    return None

def get_decimals():
    return 0 # no decimal point for arc20

@dataclass
class BalanceQuery:
    address: str
    atomical_id: "bytes | None"
    block_height: "int | None"

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

    def _resolve_ticker_to_atomical_id(self, ticker: str) -> bytes | None:
        bp = self.session_mgr.bp
        height = bp.height
        ticker = ticker.lower() # tickers are case-insensitive
        status, candidate_atomical_id, _ = bp.get_effective_ticker(ticker, height)
        if status == "verified":
            return candidate_atomical_id
        else:
            return None

    # check if this is atomicalId or ticker
    def _parse_request_id(self, id) -> bytes | None:
        if not id:
            return None
        atomical_id = None
        try:
            atomical_id = compact_to_location_id_bytes(id)
        except:
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
        except:
            pk_bytes = None
        if pk_bytes:
            addr_from_pk = get_address_from_output_script(pk_bytes)
            if addr_from_pk:
                addr = addr_from_pk
        else:
            addr = wallet
        return addr
    
    # check if block height is valid
    def _parse_block_height(self, block_height_str: str) -> int | None:
        # empty string
        if not block_height_str:
            return None
        # some character is not 0-9
        if not block_height_str.isnumeric():
            return None
        block_height = int(block_height_str)
        return block_height
    
    def _block_height_to_unix_timestamp(self, block_height: int) -> int:
        db_block_ts = self.session_mgr.db.get_block_timestamp(block_height)
        if db_block_ts:
            return db_block_ts
        return 0

    @error_handler
    async def get_block_height(self, request: "Request") -> "Response":
        block_height = self.session_mgr.db.db_height
        block_hash = self.session_mgr.db.get_atomicals_block_hash(block_height)
        return format_response({
            "hash": block_hash,
            "height": block_height,
        })

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
        conf = [{"tx_hash": hash_to_hex_str(tx_hash), "height": height}
                for tx_hash, height in history]
        return conf

    async def _get_populated_arc20_balances(self, address: str, atomical_id: "bytes | None", block_height: int):
        pk_scriptb = get_script_from_address(address)

        balances: "dict[bytes, int]" = {} # atomical_id -> amount (int)
        script_hash = sha256(pk_scriptb)
        hashX = scripthash_to_hashX(script_hash)
        if not hashX:
            raise Exception("Invalid hashX") # should not happen since we are using sha256
        history_data = await self._confirmed_history(hashX)
        # only use transactions after ATOMICALS_ACTIVATION_HEIGHT and before block_height
        history_data = [x for x in history_data if self.session_mgr.env.coin.ATOMICALS_ACTIVATION_HEIGHT <= x["height"] and x["height"] <= block_height]
        
        history_list = []
        for history in list(history_data):
            tx_num, _ = self.session_mgr.db.get_tx_num_height_from_tx_hash(hex_str_to_hash(history["tx_hash"]))
            history["tx_num"] = tx_num
            history_list.append(history)

        history_list.sort(key=lambda x: x["tx_num"])
        tx_datas = await asyncio.gather(*[self.session_mgr.get_transaction_detail(history["tx_hash"], history["height"], history["tx_num"]) for history in history_list])
        for tx_data in tx_datas:
            self._process_balance(address, balances, tx_data)
        
        # clear empty balances and filter by atomical_id
        balances = { k: v for k, v in balances.items() if v != 0 and (not atomical_id or k == atomical_id )}
        # populate atomical objects
        atomical_ids = list(balances.keys())
        atomicals_list = await asyncio.gather(*[self._get_atomical(atomical_id) for atomical_id in atomical_ids])
        atomicals: dict[bytes, dict] = {atomical_id: atomical for atomical_id, atomical in zip(atomical_ids, atomicals_list)}

        populated_balances: "list[dict]" = [] # atomical_id -> { "amount": int, "ticker": str | None }
        for atomical_id, amount in balances.items():
            atomical = atomicals.get(atomical_id)
            if atomical:
                atomical: dict = await self.session_mgr.db.populate_extended_atomical_holder_info(atomical_id, atomical)
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
            block_height = self._parse_block_height(q_block_height)
            if block_height is None:
                return format_response(None, 400, "Invalid block height.")
        
        # parse atomical_id
        id = request.query.get("id")
        atomical_id = None
        if id:
            atomical_id = self._parse_request_id(id)
            if not atomical_id:
                return format_response(None, 400, "Invalid ID.")
        
        populated_balances = await self._get_populated_arc20_balances(address, atomical_id, block_height)
        return format_response({
            "blockHeight": block_height,
            "list": populated_balances
        })
    
    @error_handler
    async def get_arc20_balances_batch(self, request: "Request") -> "Response":
        body: dict = await request.json()
        queries: "list[BalanceQuery]" = []
        latest_block_height = self.session_mgr.db.db_height

        raw_queries: "list[dict]" = body.get("queries", [])
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
                block_height = self._parse_block_height(str(q_block_height))
                if block_height is None:
                    return format_response(None, 400, f"query index {idx}: invalid block height.")
            
            # parse atomical_id
            id = raw_query.get("id")
            atomical_id = None
            if id:
                atomical_id = self._parse_request_id(id)
                if not atomical_id:
                    return format_response(None, 400, f"query index {idx}: invalid ID.")
            
            # append to list
            queries.append(BalanceQuery(address, atomical_id, block_height))
        
        results = await asyncio.gather(*[self._get_populated_arc20_balances(query.address, query.atomical_id, query.block_height) for query in queries])
        formatted_results = [{
            "blockHeight": query.block_height,
            "list": result
        } for query, result in zip(queries, results)]
        return format_response({
            "list": formatted_results
        })
    
    async def _get_tx_detail(self, tx_hash: str, f_atomical_id: bytes | None, f_address: str | None) -> dict | None:
        tx_data = await self.session_mgr.get_transaction_detail(tx_hash)
        block_height = tx_data.get("height", 0)
        tx_num = tx_data.get("tx_num", 0)
        tx_info: dict = tx_data.get("info", {})
        tx_transfers: dict = tx_data.get("transfers", {})
        tx_op = tx_data.get("op", "")
        
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
        mint_ticker = tx_payload_args.get("mint_ticker", "")
        if mint_ticker:
            tx_info_outputs: dict = tx_info.get("outputs", {})
            mint_outputs: list[dict] = tx_info_outputs.get(0, []) # mint output should be in index 0
            for output_data in mint_outputs:
                atomica_id_str = output_data.get("atomical_id", "")
                address = output_data.get("address", "")
                mint_amount = output_data.get("value", 0)
                pk_script = ""
                if address:
                    pk_script = get_script_from_address(address).hex()
                output_map = {
                    "index": output_data.get("index", ""),
                    "id": atomica_id_str,
                    "amount": str(mint_amount),
                    "decimals": get_decimals(),
                    "address": address,
                    "pkScript": pk_script,
                }
                outputs.append(output_map)
                prev_mint: dict | None = mints.get("atomica_id_str", None)
                if not prev_mint:
                    prev_mint = {
                        "amount": "0",
                        "decimals": get_decimals(),
                    }
                new_mint = {
                    "amount": str(int(prev_mint["amount"])+mint_amount),
                    "decimals": get_decimals(),
                }
                mints[atomica_id_str] = new_mint

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
            if not(found_input or found_output):
                return None
        
        if f_atomical_id:
            f_atomical_id_str = location_id_bytes_to_compact(f_atomical_id)
            found_input = f_atomical_id_str in [e["id"] for e in inputs]
            fount_output = f_atomical_id_str in [e["id"] for e in outputs]
            # TODO: more place to check?
            if not(found_input or fount_output):
                return None

        return {
            "txHash": tx_hash,
            "blockHeight": block_height,
            "index": tx_num,
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
    
    @error_handler
    async def get_arc20_transactions(self, request: "Request") -> "Response":
        # parse wallet (optional)
        wallet = request.query.get("wallet")
        address = None
        if wallet is not None:
            address = self._parse_addr(wallet)
            if not address:
                return format_response(None, 400, "Invalid wallet.")

        # parse block_height
        q_block_height = request.query.get("blockHeight")
        block_height = None
        if q_block_height is not None:
            block_height = self._parse_block_height(q_block_height)
            if block_height is None:
                return format_response(None, 400, "Invalid block height.")
        
        # parse atomical_id
        id = request.query.get("id")
        atomical_id = None
        if id:
            atomical_id = self._parse_request_id(id)
            if not atomical_id:
                return format_response(None, 400, "Invalid ID.")
            
        # no parameters passed, default to filter by latest_block_height
        if not (address or atomical_id or block_height):
            latest_block_height = self.session_mgr.db.db_height
            block_height = latest_block_height

        txs = []
        tx_hashes = []

        # if has block_height filter, use this way first
        if block_height:
            # get all tx in single block_height
            tx_hashes = self.session_mgr.db.get_atomicals_block_txs(block_height)
        
        # no block_height found, use more exhausive search
        elif atomical_id:
            # get all tx filter by id
            reverse = False
            hashX = double_sha256(atomical_id)
            history_data, _ = await self.session_mgr.get_history_op(hashX, -1, 0, None, reverse)
            for history in history_data:
                tx_hash, _ = self.session_mgr.db.fs_tx_hash(history["tx_num"])
                tx_hashes.append(hash_to_hex_str(tx_hash))
            
            # use atomical_id = None to skip filtering check
            atomical_id = None
        
        # get all tx filter by wallet
        else:
            # TODO
            tx_hashes = []
            
            # use address = None to skip filtering check
            address = None
        
        txs = await asyncio.gather(*[self._get_tx_detail(tx_hash, atomical_id, address) for tx_hash in tx_hashes])
        # filter None out
        res_txs = []
        for tx in txs:
            if tx:
                res_txs.append(tx)

        return format_response({
            "list": res_txs,
        })
    
    async def _get_arc20_holders_by_block_height(self, atomical_id: bytes, block_height: int) -> dict:
        utxos = await self.session_mgr.db.get_utxos_at_height_by_atomical_id(atomical_id, block_height)
        
        total_value = 0
        holder_map: dict[bytes, int] = {}
        
        # group by pk_script and map to address
        for utxo in utxos:
            tx_id_str = hash_to_hex_str(utxo.tx_hash)
            location = compact_to_location_id_bytes(tx_id_str + "i" + str(utxo.tx_pos))
            pk_scriptb = self.session_mgr.bp.get_pk_script_at_location(location)
            prev_value = holder_map.get(pk_scriptb, 0)
            holder_map[pk_scriptb] = prev_value + utxo.value
            total_value = total_value + utxo.value

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
        atomical_id = self._parse_request_id(id)
        if not atomical_id:
            return format_response(None, 400, "Invalid ID.")
            
        # parse block_height
        latest_block_height = self.session_mgr.db.db_height
        q_block_height = request.query.get("blockHeight")
        block_height = latest_block_height
        if q_block_height is not None:
            block_height = self._parse_block_height(q_block_height)
            if block_height is None:
                return format_response(None, 400, "Invalid block height.")

        # base data
        atomical = await self._get_atomical(atomical_id)

        atomical_type = atomical.get("type", "")
        subtype = atomical.get("subtype", "")
        mint_mode = atomical.get("$mint_mode", "")
        mint_info: dict = atomical.get("mint_info", {})
        mint_info_args: dict = mint_info.get("args", {})

        formatted_results = []
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
        elif atomical_type == "NFT":
            mint_mode = ""

        if subtype == "decentralized":
            atomical: dict = await self.session_mgr.bp.get_dft_mint_info_rpc_format_by_atomical_id(atomical_id)
            mint_count = atomical["dft_info"]["mint_count"]
            minted_amount = mint_count * mint_amount # total minted
        elif subtype == "direct":
            atomical: dict = await self.session_mgr.bp.get_ft_mint_info_rpc_format_by_atomical_id(atomical_id)
            minted_amount = max_supply # entire mint in direct mint
        
        if block_height == latest_block_height:
            atomical: dict = await self.session_mgr.db.populate_extended_atomical_holder_info(atomical_id, atomical)
            if atomical["type"] == "FT":
                for holder in atomical.get("holders", []):
                    percent = holder["holding"] / max_supply
                    formatted_results.append({
                        "address": get_address_from_output_script(bytes.fromhex(holder["script"])),
                        "pkScript": holder["script"],
                        "amount": str(holder["holding"]),
                        "percent": percent,
                    })
            elif atomical["type"] == "NFT":
                for holder in atomical.get("holders", []):
                    formatted_results.append({
                        "address": get_address_from_output_script(bytes.fromhex(holder["script"])),
                        "pkScript": holder["script"],
                        "amount": str(holder["holding"]),
                        "percent": 1,
                    })
        else:
            # support only atomical FT
            data = await self._get_arc20_holders_by_block_height(atomical_id, block_height)
            for pk_scriptb, amount in data.get("holders", {}).items():
                formatted_results.append({
                    "address": get_address_from_output_script(pk_scriptb),
                    "pkScript": pk_scriptb.hex(),
                    "amount": str(amount),
                    "percent": amount / max_supply,
                })

        # sort by holding desc
        formatted_results.sort(key=lambda x: (int(x["amount"]), x["address"]), reverse=True)

        return format_response({
            "blockHeight": block_height,
            "totalSupply": str(max_supply),
            "mintedAmount": str(minted_amount),
            "decimals": get_decimals(),
            "list": formatted_results
        })
    
    @error_handler
    async def get_arc20_token(self, request: "Request") -> "Response":
        # parse atomical_id
        id = request.match_info.get("id", "")
        if not id:
            return format_response(None, 400, "ID is required.")
        atomical_id = self._parse_request_id(id)
        if not atomical_id:
            return format_response(None, 400, "Invalid ID.")
        
        # parse block_height
        latest_block_height = self.session_mgr.db.db_height
        q_block_height = request.query.get("blockHeight")
        block_height = latest_block_height
        if q_block_height is not None:
            block_height = self._parse_block_height(q_block_height)
            if block_height is None:
                return format_response(None, 400, "Invalid block height.")
        
        # get data
        atomical: dict = await self._get_atomical(atomical_id)

        atomical_type = atomical.get("type", "")
        subtype = atomical.get("subtype", "")
        mint_mode = atomical.get("$mint_mode", "")
        mint_info: dict = atomical.get("mint_info", {})
        mint_info_args: dict = mint_info.get("args", {})
        ticker = atomical.get("$ticker", "")

        mint_count = 0
        minted_amount = 0
        max_supply = 0
        mint_amount = 0 # mint size

        if atomical_type == "FT":
            if mint_mode == "fixed":
                max_supply = atomical.get("$max_supply", 0)
                mint_amount = mint_info_args.get("mint_amount", 0)
            else:
                max_supply = atomical.get("$max_supply", -1)
                if max_supply < 0:
                    mint_amount = mint_info_args.get("mint_amount", 0)
                    max_supply = DFT_MINT_MAX_MAX_COUNT_DENSITY * mint_amount
        elif atomical_type == "NFT":
            mint_mode = ""

        if subtype == "decentralized":
            atomical: dict = await self.session_mgr.bp.get_dft_mint_info_rpc_format_by_atomical_id(atomical_id)
            mint_count = atomical["dft_info"]["mint_count"]
            minted_amount = mint_count * mint_amount # total minted
        elif subtype == "direct":
            atomical: dict = await self.session_mgr.bp.get_ft_mint_info_rpc_format_by_atomical_id(atomical_id)
            minted_amount = max_supply # entire mint in direct mint

        location_summary: dict = atomical.get("location_summary", {})
        holder_count = location_summary.get("unique_holders", 0)
        circulating_supply = location_summary.get("circulating_supply", 0)

        # deployment data
        commit_tx_id = mint_info.get("commit_txid")
        commit_tx_height = mint_info.get("commit_height")

        reveal_location_script: str = mint_info.get("reveal_location_script")
        deployer_address = get_address_from_output_script(bytes.fromhex(reveal_location_script))

        deployed_at_height = commit_tx_height
        deployed_at = self._block_height_to_unix_timestamp(deployed_at_height)
        deploy_tx_hash = commit_tx_id

        # mint completion data
        completed_at_height = None
        completed_at = None # unix timestamp
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

        # change time-sensitive data from block_height
        if block_height != latest_block_height:
            # support only atomical FT
            data = await self._get_arc20_holders_by_block_height(atomical_id, block_height)
            holder_count = data["count"]
            circulating_supply = data["total"]
            if subtype == "decentralized":
                mint_count = await self.session_mgr.db.get_atomical_mint_count_at_height(atomical_id, block_height)
                minted_amount = mint_count * mint_amount

        compact_atomical_id = location_id_bytes_to_compact(atomical_id)
        return format_response({
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
                "maxMints": str(atomical.get("$max_mints", 0)), # number of times this token can be minted
                "deployedBy": deployer_address,
                "mintHeight": atomical.get("$mint_height", 0), # the block height this FT can start to be minted
                "mintInfo": {
                    "commitTxHash": commit_tx_id,
                    "commitIndex": mint_info.get("commit_index"), # commit tx output index of utxo used in reveal tx
                    "revealTxHash": mint_info.get("reveal_location_txid"),
                    "revealIndex": mint_info.get("reveal_location_index"),
                    "args": mint_info_args, # raw atomicals operation payload
                    "metadata": mint_info.get("meta", {}) # metadata.json used during deployment
                },
                "subtype": subtype,
                "mintMode": mint_mode
            }
        })
    
    async def _utxo_to_formatted(self, utxo: UTXO) -> "dict":
        tx_id_str = hash_to_hex_str(utxo.tx_hash)
        output_index = utxo.tx_pos
        location = compact_to_location_id_bytes(tx_id_str + "i" + str(output_index))
        atomical_by_location = self.session_mgr.db.get_atomicals_by_location_extended_info_long_form(location)
        location_info: dict = atomical_by_location.get("location_info", {})
        atomical_amount = location_info.get("value", 0)
        atomical_ids: list = atomical_by_location.get("atomicals", [])
        
        formatted_atomicals = []
        # should be 0 or 1 item
        for atomical_id in atomical_ids:
            atomical = await self._get_atomical(atomical_id)
            ticker = atomical.get("$ticker", "")
            atomical_out = {
                "atomicalId": location_id_bytes_to_compact(atomical_id),
                "ticker": ticker,
                "amount": str(atomical_amount),
                "decimals": get_decimals(),
            }
            formatted_atomicals.append(atomical_out)
        res = {
            "txHash": tx_id_str,
            "outputIndex": utxo.tx_pos,
            "sats": utxo.value,
            "extend": {
                "atomicals": formatted_atomicals
            }
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
            block_height = self._parse_block_height(q_block_height)
            if block_height is None:
                return format_response(None, 400, "Invalid block height.")
        
        # parse atomical_id
        id = request.query.get("id")
        atomical_id = None
        if id:
            atomical_id = self._parse_request_id(id)
            if not atomical_id:
                return format_response(None, 400, "Invalid ID.")
        
        formatted_results: list[dict] = []
        utxos: list[UTXO] = []
        
        # bypass for latest_block
        if block_height == latest_block_height:
            hashX = scripthash_to_hashX(sha256(pk_scriptb))
            utxos = await self.session_mgr.db.all_utxos(hashX)
        else:
            utxos = await self.session_mgr.db.get_utxos_at_height_by_pk_script(pk_scriptb, block_height)
        
        formatted_results = await asyncio.gather(*[self._utxo_to_formatted(utxo) for utxo in utxos])
        
        # filter by atomical_id
        if atomical_id:
            atomical_id_str = location_id_bytes_to_compact(atomical_id)
        
        # return only UTXOs that contain atomical, or filter if parameter passed
        filtered_formatted = []
        for e in formatted_results:
            atomical_list: list = e["extend"]["atomicals"]
            # skip UTXO that not contain atomical
            if len(atomical_list) == 0:
                continue
            found = True
            if atomical_id:
                found = atomical_id_str in [a["atomicalId"] for a in atomical_list]
            if found:
                filtered_formatted.append(e)

        return format_response({
            "blockHeight": block_height,
            "list": filtered_formatted,
        })
