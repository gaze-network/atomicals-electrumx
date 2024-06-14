import asyncio
import base64
from dataclasses import dataclass
import json
from typing import TYPE_CHECKING, Optional
from aiohttp.web import json_response
from aiohttp.web_urldispatcher import UrlDispatcher

from electrumx.lib import util
from electrumx.lib.hash import HASHX_LEN, hex_str_to_hash, sha256
from electrumx.lib.script2addr import get_address_from_output_script, get_script_from_address
from electrumx.lib.util_atomicals import DFT_MINT_MAX_MAX_COUNT_DENSITY, compact_to_location_id_bytes, location_id_bytes_to_compact
from electrumx.server.block_processor import BlockProcessor
from electrumx.server.db import DB
from electrumx.server.http_session import HttpHandler

if TYPE_CHECKING:
    from electrumx.server.env import Env
    from electrumx.server.controller import SessionManager
    from aiohttp.web import Request, Response

class JSONBytesEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, bytes):
            # use base64 encoding for bytes
            return base64.b64encode(obj).decode()
        return super().default(obj)

def format_response(result: 'dict | None', status: 'int | None' = None, error: 'str | None' = None) -> 'Response':
    if error:
        if status == None:
            status = 500
        return json_response({
                'error': error,
            },
            status=status,
            dumps=lambda o: json.dumps(o, cls=JSONBytesEncoder)
        )
    if status == None:
        status = 200
    return json_response({
                'error': None,
                'result': result,
            },
            status=status,
            dumps=lambda o: json.dumps(o, cls=JSONBytesEncoder)
        )

def scripthash_to_hashX(script_hash: bytes) -> 'Optional[bytes]':
    if len(script_hash) == 32:
        return script_hash[:HASHX_LEN]
    return None

@dataclass
class BalanceQuery:
    address: str
    atomical_id: 'bytes | None'
    block_height: 'int | None'

class HttpUnifiedAPIHandler(object):
    def __init__(self, session_mgr: 'SessionManager', env: 'Env', db: DB, bp: BlockProcessor, http_handler: HttpHandler):
        self.logger = util.class_logger(__name__, self.__class__.__name__)
        self.env = env
        self.db = db
        self.bp = bp
        self.session_mgr = session_mgr
        self.http_handler = http_handler

    def error_handler(func):
        async def wrapper(self: 'HttpUnifiedAPIHandler', *args, **kwargs):
            try:
                return await func(self, *args, **kwargs)
            except Exception as e:
                self.logger.exception(f'Request has failed with exception: {repr(e)}')
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
        if status == 'verified':
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
                raise Exception("Ticker is not found or is not confirmed.")
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
            addr_from_pk = get_address_from_output_script(bytes.fromhex(wallet))
            if addr_from_pk:
                addr = addr_from_pk
        else:
            addr = wallet
        return addr

    @error_handler
    async def get_block_height(self, request: 'Request') -> 'Response':
        block_height = self.db.db_height
        block_hash = self.db.get_atomicals_block_hash(block_height)
        return format_response({
            "hash": block_hash,
            "height": block_height,
        })

    def _process_balance(self, address: str, balances: 'dict[bytes, int]', tx_data):
        inputs = tx_data['transfers']['inputs']
        outputs = tx_data['transfers']['outputs']
        for _, input_atomicals in inputs.items():
            for input_atomical in input_atomicals:
                if input_atomical['address'] == address and input_atomical['type'] == 'FT':
                    atomical_id_str = input_atomical['atomical_id']
                    atomical_id = compact_to_location_id_bytes(atomical_id_str)
                    if atomical_id not in balances:
                        balances[atomical_id] = 0
                    balances[atomical_id] -= input_atomical['value']
        for _, output_atomicals in outputs.items():
            for output_atomical in output_atomicals:
                if output_atomical['address'] == address and output_atomical['type'] == 'FT':
                    atomical_id_str = output_atomical['atomical_id']
                    atomical_id = compact_to_location_id_bytes(atomical_id_str)
                    if atomical_id not in balances:
                        balances[atomical_id] = 0
                    balances[atomical_id] += output_atomical['value']
        if tx_data['op'] in ['mint-dft', 'mint-ft']:
            # minted fts is always at output index 0
            minted_fts = tx_data['info']['outputs'][0]
            for minted_ft in minted_fts:
                atomical_id_str = minted_ft['atomical_id']
                atomical_id = compact_to_location_id_bytes(atomical_id_str)
                if atomical_id not in balances:
                    balances[atomical_id] = 0
                balances[atomical_id] += minted_ft['value']

    async def _get_atomical(self, atomical_id: bytes) -> 'Optional[str]':
        compact_atomical_id = location_id_bytes_to_compact(atomical_id)
        atomical = await self.http_handler.atomical_id_get(compact_atomical_id)
        return atomical

    async def _get_populated_arc20_balances(self, address: str, atomical_id: 'bytes | None', block_height: int):
        pk_scriptb = get_script_from_address(address)

        balances: 'dict[bytes, int]' = {} # atomical_id -> amount (int)
        script_hash = sha256(pk_scriptb)
        hashX = scripthash_to_hashX(script_hash)
        if not hashX:
            raise Exception('Invalid hashX') # should not happen since we are using sha256
        history_data = await self.http_handler.confirmed_history(hashX)
        # only use transactions after ATOMICALS_ACTIVATION_HEIGHT and before block_height
        history_data = [x for x in history_data if self.env.coin.ATOMICALS_ACTIVATION_HEIGHT <= x["height"] and x["height"] <= block_height]
        
        history_list = []
        for history in list(history_data):
            tx_num, _ = self.db.get_tx_num_height_from_tx_hash(hex_str_to_hash(history["tx_hash"]))
            history['tx_num'] = tx_num
            history_list.append(history)

        history_list.sort(key=lambda x: x['tx_num'])
        tx_datas = await asyncio.gather(*[self.session_mgr.get_transaction_detail(history["tx_hash"], history["height"], history["tx_num"]) for history in history_list])
        for tx_data in tx_datas:
            self._process_balance(address, balances, tx_data)
        
        # clear empty balances and filter by atomical_id
        balances = { k: v for k, v in balances.items() if v != 0 and (not atomical_id or k == atomical_id )}
        # populate atomical objects
        atomical_ids = list(balances.keys())
        atomicals_list = await asyncio.gather(*[self._get_atomical(atomical_id) for atomical_id in atomical_ids])
        atomicals = {atomical_id: atomical for atomical_id, atomical in zip(atomical_ids, atomicals_list)}

        populated_balances: 'list[dict]' = [] # atomical_id -> { "amount": int, "ticker": str | None }
        for atomical_id, amount in balances.items():
            atomical = atomicals.get(atomical_id)
            if atomical:
                atomical = await self.db.populate_extended_atomical_holder_info(atomical_id, atomical)
                ticker = atomical.get("ticker")
                if not ticker:
                    ticker = "" # default to empty string
                balance = {
                    "amount": str(amount),
                    "id": location_id_bytes_to_compact(atomical_id),
                    "name": ticker,
                    "symbol": ticker,
                    "decimals": 0, # no decimal point for arc20
                }
                populated_balances.append(balance)
        return populated_balances
    
    @error_handler
    async def get_arc20_balance(self, request: 'Request') -> 'Response':
        # parse wallet
        wallet = request.match_info.get("wallet", "")
        address = self._parse_addr(wallet)
        if not address:
            return format_response(None, 400, 'Invalid wallet.')

        # parse block_height
        latest_block_height = self.db.db_height
        block_height = int(request.query.get('blockHeight', latest_block_height))
        
        # parse atomical_id
        id = request.query.get("id")
        atomical_id = None
        if id:
            try:
                atomical_id = self._parse_request_id(id)
            except:
                return format_response(None, 400, 'Invalid ID.')

        populated_balances = await self._get_populated_arc20_balances(address, atomical_id, block_height)
        return format_response({
            "blockHeight": block_height,
            "list": populated_balances
        })
    
    @error_handler
    async def get_arc20_balances_batch(self, request: 'Request') -> 'Response':
        body = await request.json()
        queries: 'list[BalanceQuery]' = []
        latest_block_height = self.db.db_height

        raw_queries: 'list[dict]' = body.get('queries', [])
        for idx, raw_query in enumerate(raw_queries):
            # parse wallet
            wallet = raw_query.get("wallet", "")
            address = self._parse_addr(wallet)
            if not address:
                return format_response(None, 400, f'query index {idx}: invalid wallet.')
            
            # parse block_height
            block_height = int(raw_query.get('blockHeight', latest_block_height))
            
            # parse atomical_id
            id = raw_query.get("id")
            atomical_id = None
            if id:
                try:
                    atomical_id = self._parse_request_id(id)
                except:
                    return format_response(None, 400, f'query index {idx}: invalid ID.')
            
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
    
    @error_handler
    async def get_arc20_transactions(self, request: 'Request') -> 'Response':
        # parse wallet (optional)
        wallet = request.query.get("wallet", "")
        address = self._parse_addr(wallet)

        # parse block_height
        latest_block_height = self.db.db_height
        block_height = int(request.query.get('blockHeight', latest_block_height))
        
        # parse atomical_id
        id = request.query.get("id")
        atomical_id = None
        if id:
            try:
                atomical_id = self._parse_request_id(id)
            except:
                return format_response(None, 400, 'Invalid ID.')
        return format_response(None, 500, "impl")
    
    @error_handler
    async def get_arc20_holders(self, request: 'Request') -> 'Response':
        # parse atomical_id
        id = request.match_info.get("id", "")
        atomical_id = None
        if id:
            try:
                atomical_id = self._parse_request_id(id)
            except:
                return format_response(None, 400, 'Invalid ID.')
            
        # parse block_height
        latest_block_height = self.db.db_height
        block_height = int(request.query.get('blockHeight', latest_block_height))

        # TODO: add block_height to filter

        atomical = await self._get_atomical(atomical_id)
        atomical = await self.db.populate_extended_atomical_holder_info(atomical_id, atomical)
        formatted_results = []
        max_supply = 0
        mint_amount = 0
        if atomical["type"] == "FT":
            if atomical["$mint_mode"] == "fixed":
                max_supply = atomical.get('$max_supply', 0)
            else:
                max_supply = atomical.get('$max_supply', -1)
                if max_supply < 0:
                    mint_amount = atomical.get("mint_info", {}).get("args", {}).get("mint_amount")
                    max_supply = DFT_MINT_MAX_MAX_COUNT_DENSITY * mint_amount
            for holder in atomical.get("holders", []):
                percent = holder['holding'] / max_supply
                formatted_results.append({
                    "address": get_address_from_output_script(bytes.fromhex(holder['script'])),
                    "pkScript": holder['script'],
                    "percent": percent,
                    "holding": holder["holding"]
                })
        elif atomical["type"] == "NFT":
            for holder in atomical.get("holders", []):
                formatted_results.append({
                    "address": get_address_from_output_script(bytes.fromhex(holder['script'])),
                    "pkScript": holder['script'],
                    "percent": 1,
                    "holding": holder["holding"]
                })
        
        # sort by holding desc
        formatted_results.sort(key=lambda x: x['holding'], reverse=True)

        return format_response({
            "blockHeight": block_height,
            "totalSupply": str(max_supply),
            "mintedAmount": str(mint_amount),
            "list": formatted_results
        })
    
    @error_handler
    async def get_arc20_token(self, request: 'Request') -> 'Response':
        # parse atomical_id
        id = request.match_info.get("id", "")
        atomical_id = None
        if id:
            try:
                atomical_id = self._parse_request_id(id)
            except:
                return format_response(None, 400, 'Invalid ID.')
        return format_response(None, 500, "impl")
    
    @error_handler
    async def get_arc20_utxos(self, request: 'Request') -> 'Response':
        # parse wallet
        wallet = request.match_info.get("wallet", "")
        address = self._parse_addr(wallet)
        
        return format_response(None, 500, "impl")
