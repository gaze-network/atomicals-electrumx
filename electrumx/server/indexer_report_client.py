import json
import requests
import electrumx
from electrumx.lib.util import class_logger
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from electrumx.server.env import Env
import socket
import requests.packages.urllib3.util.connection as urllib3_cn
    

# patch requests package to use ipv4 only for client ip address
def allowed_gai_family():
    """
    https://github.com/urllib3/urllib3/blob/main/src/urllib3/util/connection.py
    """
    family = socket.AF_INET
    if urllib3_cn.HAS_IPV6:
        family = socket.AF_INET6 # force ipv6 only if it is available
    return family

urllib3_cn.allowed_gai_family = allowed_gai_family


class IndexerReportClient:
    def __init__(self, env: 'Env'):
        self.logger = class_logger(__name__, self.__class__.__name__)
        self.env = env
    
    def submit_block_report(self, db_version: int, type: str, height: int, block_hash: bytes, event_hash: bytes, cumulative_event_hash: bytes):
        data = {
            'type': type,
            'clientVersion': electrumx.indexer_version,
            'dbVersion': db_version,
            'network': self.env.coin.NET,
            'blockHeight': height,
            'blockHash': block_hash.hex(),
            'eventHash': event_hash.hex(),
            'cumulativeEventHash': cumulative_event_hash.hex(),
        }
        resp = requests.post(f'{self.env.indexer_report.url}/v1/report/block', json=data)
        if resp.status_code >= 400:
            self.logger.warning(f'submit_block_report: Failed to submit block report. request_body={json.dumps(data)} status_code={resp.status_code}, text={resp.text}')
        else:
            self.logger.info(f"submit_block_report: Submitted block report to indexer: db_version {db_version} type {type} height {height}, block_hash {block_hash.hex()}, event_hash {event_hash.hex()}, cumulative_event_hash {cumulative_event_hash.hex()}")
    
    def submit_node_report(self, type: str):
        data = {
            'name': self.env.indexer_report.name,
            'type': type,
            'network': self.env.coin.NET,
            'websiteURL': self.env.indexer_report.website_url,
            'indexerAPIURL': self.env.indexer_report.indexer_api_url,
        }
        resp = requests.post(f'{self.env.indexer_report.url}/v1/report/node', json=data)
        if resp.status_code >= 400:
            self.logger.warning(f'submit_node_report: Failed to submit node report. request_body={json.dumps(data)}, status_code={resp.status_code}, text={resp.text}')
        else:
            self.logger.info(f'submit_node_report: Submitted node report. name={data["name"]}, type={data["type"]}, websiteURL={data["websiteURL"]}, indexerAPIURL={data["indexerAPIURL"]}')
