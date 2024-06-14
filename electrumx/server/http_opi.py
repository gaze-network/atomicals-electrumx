import asyncio
import base64
from dataclasses import dataclass
import json
from electrumx.lib.hash import HASHX_LEN, hex_str_to_hash, sha256
from electrumx.lib.script2addr import get_address_from_output_script, get_script_from_address
from electrumx.lib.util_atomicals import DFT_MINT_MAX_MAX_COUNT_DENSITY, compact_to_location_id_bytes, location_id_bytes_to_compact
from electrumx.server.block_processor import BlockProcessor
from electrumx.server.db import DB
from electrumx.server.http_session import HttpHandler, scripthash_to_hashX
import electrumx.lib.util as util
from aiohttp.web_urldispatcher import UrlDispatcher
from typing import TYPE_CHECKING
from aiohttp.web import json_response
if TYPE_CHECKING:
    from electrumx.server.env import Env
    from typing import Optional
    from electrumx.server.session import SessionManager
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

def tx_contains_atomical_id(tx_data, atomical_id):
    atomical_id_str = location_id_bytes_to_compact(atomical_id)
    for _, input_atomicals in tx_data['transfers']['inputs'].items():
        for input_atomical in input_atomicals:
            if input_atomical['atomical_id'] == atomical_id_str:
                return True
    for _, output_atomicals in tx_data['transfers']['outputs'].items():
        for output_atomical in output_atomicals:
            if output_atomical['atomical_id'] == atomical_id_str:
                return True
    if atomical_id_str in tx_data['transfers']['burned_fts'] and tx_data['transfers']['burned_fts'][atomical_id] > 0:
        return True
    if tx_data['info'].get('atomical_id') == atomical_id_str:
        return True
    payment = tx_data['info'].get('payment')
    if payment and payment.get('atomical_id') == atomical_id_str:
        return True
    return False

@dataclass
class BalanceQuery:
    address: 'str | None'
    atomical_id: 'bytes | None'
    block_height: 'int | None'

class HttpOPIHandler(object):
    def __init__(self, session_mgr: 'SessionManager', env: 'Env', db: DB, bp: BlockProcessor, http_handler: HttpHandler):
        self.logger = util.class_logger(__name__, self.__class__.__name__)
        self.env = env
        self.db = db
        self.bp = bp
        self.session_mgr = session_mgr
        self.http_handler = http_handler

    def error_handler(func):
        async def wrapper(self: 'HttpOPIHandler', *args, **kwargs):
            try:
                return await func(self, *args, **kwargs)
            except Exception as e:
                self.logger.exception(f'Request has failed with exception: {repr(e)}')
                return format_response(None, 500, "Internal Server Error")
        return wrapper

    def mount_routes(self, router: UrlDispatcher):
        router.add_get('/v1/arc20/block_height', self.get_block_height)
        router.add_get('/v1/arc20/balance', self.get_arc20_balance)
        router.add_post('/v1/arc20/balances', self.get_arc20_balances_batch)
        router.add_get('/v1/arc20/activity', self.get_arc20_activity)
        router.add_get('/v1/arc20/holders', self.get_arc20_holders)
    
    def _resolve_ticker_to_atomical_id(self, ticker: str) -> bytes:
        height = self.bp.height
        ticker = ticker.lower() # tickers are case-insensitive
        status, candidate_atomical_id, _ = self.bp.get_effective_ticker(ticker, height)
        if status == 'verified':
            return candidate_atomical_id
        else:
            return None

    @error_handler
    async def get_block_height(self, request: 'Request') -> 'Response':
        block_height = self.db.db_height
        return format_response({
            'block_height': block_height
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

        populated_balances: 'dict[str, dict]' = {} # atomical_id -> { "amount": int, "ticker": str | None }
        for atomical_id, amount in balances.items():
            atomical = atomicals.get(atomical_id)
            if atomical:
                atomical = await self.db.populate_extended_atomical_holder_info(atomical_id, atomical)
                balance = {
                    "amount": amount,
                }
                ticker = atomical.get("ticker")
                if ticker:
                    balance["ticker"] = ticker
                populated_balances[location_id_bytes_to_compact(atomical_id)] = balance
        return populated_balances
    
    @error_handler
    async def get_arc20_balance(self, request: 'Request') -> 'Response':
        compact_atomical_id = request.query.get('atomical_id')
        ticker = request.query.get('ticker')
        block_height = int(request.query.get('block_height', 0))
        address = request.query.get('address')
        pk_script = request.query.get('pk_script')

        # parse atomical_id
        atomical_id = None
        if compact_atomical_id:
            atomical_id = compact_to_location_id_bytes(compact_atomical_id)
        elif ticker:
            # if compact_atomical_id is not provided, attempt to get the atomical_id from the ticker
            atomical_id = self._resolve_ticker_to_atomical_id(ticker)
            if not atomical_id:
                return format_response(None, 400, 'Ticker is not found or is not confirmed.')
        if not block_height:
            block_height = self.db.db_height

        # parse address
        if not address:
            if not pk_script:
                return format_response(None, 400, 'Either "pk_script" or "address" must be provided')
            address = get_address_from_output_script(bytes.fromhex(pk_script))
        
        populated_balances = await self._get_populated_arc20_balances(address, atomical_id, block_height)
        return format_response({
            "balances": populated_balances
        })

    @error_handler
    async def get_arc20_balances_batch(self, request: 'Request') -> 'Response':
        body = await request.json()
        queries: 'list[BalanceQuery]' = []
        latest_block_height = self.db.db_height

        raw_queries: 'list[dict]' = body.get('queries', [])
        for idx, raw_query in enumerate(raw_queries):
            address = raw_query.get('address')
            pk_script = raw_query.get('pk_script')
            compact_atomical_id = raw_query.get('atomical_id')
            ticker = raw_query.get('ticker')
            block_height = int(raw_query.get('block_height', latest_block_height))

            # parse atomical_id
            atomical_id = None
            if compact_atomical_id:
                atomical_id = compact_to_location_id_bytes(compact_atomical_id)
            elif ticker:
                # if compact_atomical_id is not provided, attempt to get the atomical_id from the ticker
                atomical_id = self._resolve_ticker_to_atomical_id(ticker)
                if not atomical_id:
                    return format_response(None, 400, f'query index {idx}: ticker is not found or is not confirmed.')
                
            # parse address
            if not address:
                if not pk_script:
                    return format_response(None, 400, f'query index {idx}: either "pk_script" or "address" must be provided')
                address = get_address_from_output_script(bytes.fromhex(pk_script))
            queries.append(BalanceQuery(address, atomical_id, block_height))
        
        results = await asyncio.gather(*[self._get_populated_arc20_balances(query.address, query.atomical_id, query.block_height) for query in queries])
        def optional_pk_script(address: str):
            pk_script = get_script_from_address(address)
            return pk_script.hex() if pk_script else None
        formatted_results = [{
            "address": query.address,
            "pk_script": optional_pk_script(query.address),
            "block_height": query.block_height,
            "balances": result
        } for query, result in zip(queries, results)]
        return format_response({
            "list": formatted_results
        })
        
    
    @error_handler
    async def get_arc20_activity(self, request: 'Request') -> 'Response':
        compact_atomical_id = request.query.get('atomical_id')
        ticker = request.query.get('ticker')
        block_height = int(request.query.get('block_height', 0))
        offset = int(request.query.get('offset', 0))
        limit = int(request.query.get('limit', 100))

        atomical_id = None
        if compact_atomical_id:
            atomical_id = compact_to_location_id_bytes(compact_atomical_id)
        elif ticker:
            # if compact_atomical_id is not provided, attempt to get the atomical_id from the ticker
            atomical_id = self._resolve_ticker_to_atomical_id(ticker)
            if not atomical_id:
                return format_response(None, 400, 'Ticker is not found or is not confirmed.')
        if not block_height:
            block_height = self.db.db_height

        txs = []
        tx_hashes = self.db.get_atomicals_block_txs(block_height)
        for tx in tx_hashes:
            # get operation by db method
            tx_num, _ = self.db.get_tx_num_height_from_tx_hash(hex_str_to_hash(tx))
            txs.append({
                "tx_num": tx_num, 
                "tx_hash": tx,
                "height": block_height
            })
            
        txs.sort(key=lambda x: x['tx_num'])
        tx_datas = await asyncio.gather(*[self.session_mgr.get_transaction_detail(tx["tx_hash"], block_height, tx["tx_num"]) for tx in txs])
        result = []
        for tx_data in tx_datas:
            # filter by atomical_id if specified
            if (atomical_id and tx_contains_atomical_id(tx_data, atomical_id)) or (not atomical_id):
                result.append(tx_data)
        total = len(result)
        return format_response({
            "total": total,
            "block_height": block_height,
            "list": result[offset:offset+limit]
        })
    
    @error_handler
    async def get_arc20_holders(self, request: 'Request') -> 'Response':
        ticker = request.query.get('ticker')
        offset = int(request.query.get('offset', 0))
        limit = int(request.query.get('limit', 100))
        compact_atomical_id = request.query.get('atomical_id')
        if compact_atomical_id:
            atomical_id = compact_to_location_id_bytes(compact_atomical_id)
        elif ticker:
            # if compact_atomical_id is not provided, attempt to get the atomical_id from the ticker
            atomical_id = self._resolve_ticker_to_atomical_id(ticker)
            if not atomical_id:
                return format_response(None, 400, 'Ticker is not found or is not confirmed.')
            compact_atomical_id = location_id_bytes_to_compact(atomical_id)
        else:
            return format_response(None, 400, 'Either "ticker" or "atomical_id" must be provided')
        
        atomical = await self._get_atomical(atomical_id)
        atomical = await self.db.populate_extended_atomical_holder_info(atomical_id, atomical)
        formatted_results = []
        if atomical["type"] == "FT":
            if atomical["$mint_mode"] == "fixed":
                max_supply = atomical.get('$max_supply', 0)
            else:
                max_supply = atomical.get('$max_supply', -1)
                if max_supply < 0:
                    mint_amount = atomical.get("mint_info", {}).get("args", {}).get("mint_amount")
                    max_supply = DFT_MINT_MAX_MAX_COUNT_DENSITY * mint_amount 
            for holder in atomical.get("holders", [])[offset:offset+limit]:
                percent = holder['holding'] / max_supply
                formatted_results.append({
                    "percent": percent,
                    "address": get_address_from_output_script(bytes.fromhex(holder['script'])),
                    "holding": holder["holding"]
                })
        elif atomical["type"] == "NFT":
            for holder in atomical.get("holders", [])[offset:offset+limit]:
                formatted_results.append({
                    "address": get_address_from_output_script(bytes.fromhex(holder['script'])),
                    "holding": holder["holding"]
                })
        formatted_results.sort(key=lambda x: x['holding'], reverse=True)
        
        return format_response({
            "list": formatted_results
        })
