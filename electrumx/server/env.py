# Copyright (c) 2016, Neil Booth
#
# All rights reserved.
#
# See the file "LICENCE" for information about the copyright
# and warranty status of this software.

"""Class for handling environment configuration and defaults."""


import re
import sys
from dataclasses import dataclass
from ipaddress import IPv4Address, IPv6Address
from typing import Type, Union

from aiorpcx import Service, ServicePart

from electrumx.lib.coins import AtomicalsCoinMixin, Coin
from electrumx.lib.env_base import EnvBase


def is_test_environment():
    return "pytest" in sys.modules


class ServiceError(Exception):
    pass


class Env(EnvBase):
    """Wraps environment configuration. Optionally, accepts a Coin class
    as first argument to have ElectrumX serve custom coins not part of
    the standard distribution.
    """

    # Peer discovery
    PD_OFF, PD_SELF, PD_ON = ("OFF", "SELF", "ON")
    SSL_PROTOCOLS = {"ssl", "wss"}
    KNOWN_PROTOCOLS = {"ssl", "tcp", "ws", "wss", "rpc", "http"}

    coin: Type[Union["Coin", "AtomicalsCoinMixin"]]

    def __init__(self, coin=None):
        super().__init__()
        self.obsolete(
            [
                "MAX_SUBSCRIPTIONS",
                "MAX_SUBS",
                "MAX_SESSION_SUBS",
                "BANDWIDTH_LIMIT",
                "HOST",
                "TCP_PORT",
                "SSL_PORT",
                "RPC_HOST",
                "RPC_PORT",
                "REPORT_HOST",
                "REPORT_TCP_PORT",
                "REPORT_SSL_PORT",
                "REPORT_HOST_TOR",
                "REPORT_TCP_PORT_TOR",
                "REPORT_SSL_PORT_TOR",
            ]
        )

        # Core items

        self.db_dir = self.required("DB_DIRECTORY")
        self.daemon_url = self.required("DAEMON_URL")
        self.daemon_proxy_url = self.default("DAEMON_PROXY_URL", None)
        self.daemon_rate_limit_max_rate = self.integer("DAEMON_RATE_LIMIT_MAX_RATE", None)
        self.daemon_rate_limit_period_sec = self.integer("DAEMON_RATE_LIMIT_PERIOD_SEC", None)
        if coin is not None:
            assert issubclass(coin, Coin)
            self.coin = coin
        else:
            coin_name = self.required("COIN").strip()
            network = self.default("NET", "mainnet").strip()
            self.coin = Coin.lookup_coin_class(coin_name, network)

        # Peer discovery

        self.peer_discovery = self.peer_discovery_enum()
        self.peer_announce = self.boolean("PEER_ANNOUNCE", True)
        self.force_proxy = self.boolean("FORCE_PROXY", False)
        self.tor_proxy_host = self.default("TOR_PROXY_HOST", "localhost")
        self.tor_proxy_port = self.integer("TOR_PROXY_PORT", None)

        # Misc

        self.db_engine = self.default("DB_ENGINE", "leveldb")
        self.banner_file = self.default("BANNER_FILE", None)
        self.tor_banner_file = self.default("TOR_BANNER_FILE", self.banner_file)
        self.anon_logs = self.boolean("ANON_LOGS", False)
        self.log_sessions = self.integer("LOG_SESSIONS", 3600)
        self.log_level = self.default("LOG_LEVEL", "info").upper()
        self.donation_address = self.default("DONATION_ADDRESS", "")
        self.drop_client = self.custom("DROP_CLIENT", None, re.compile)
        self.drop_client_unknown = self.boolean("DROP_CLIENT_UNKNOWN", False)
        self.blacklist_url = self.default("BLACKLIST_URL", self.coin.BLACKLIST_URL)
        self.cache_MB = self.integer("CACHE_MB", 1200)
        self.reorg_limit = self.integer("REORG_LIMIT", self.coin.REORG_LIMIT)
        self.daemon_poll_interval_blocks_msec = self.integer("DAEMON_POLL_INTERVAL_BLOCKS", 5000)
        self.daemon_poll_interval_mempool_msec = self.integer("DAEMON_POLL_INTERVAL_MEMPOOL", 5000)

        # Server limits to help prevent DoS

        self.max_send = self.integer("MAX_SEND", self.coin.DEFAULT_MAX_SEND)
        self.max_recv = self.integer("MAX_RECV", 1_000_000)
        self.max_sessions = self.sane_max_sessions()
        self.cost_soft_limit = self.integer("COST_SOFT_LIMIT", 1000)
        self.cost_hard_limit = self.integer("COST_HARD_LIMIT", 10000)
        self.bw_unit_cost = self.integer("BANDWIDTH_UNIT_COST", 5000)
        self.initial_concurrent = self.integer("INITIAL_CONCURRENT", 10)
        self.request_sleep = self.integer("REQUEST_SLEEP", 2500)
        self.request_timeout = self.integer("REQUEST_TIMEOUT", 30)
        self.session_timeout = self.integer("SESSION_TIMEOUT", 600)
        self.session_group_by_subnet_ipv4 = self.integer("SESSION_GROUP_BY_SUBNET_IPV4", 24)
        self.session_group_by_subnet_ipv6 = self.integer("SESSION_GROUP_BY_SUBNET_IPV6", 48)
        # aiohttp.web_app.Application.client_max_size
        self.session_max_size_http = self.integer("SESSION_MAX_SIZE_HTTP", 1024**2)
        # websockets.legacy.server.Serve.max_size
        self.session_max_size_ws = self.integer("SESSION_MAX_SIZE_WS", 1024**2)
        self._check_and_fix_cost_limits()
        self.enable_rate_limit = self.boolean("ENABLE_RATE_LIMIT", True)

        # Indexer reporting system
        self.gaze_network_report = self.set_gaze_network_report_config()

        # Services last - uses some env vars above

        self.services = self.services_to_run()
        if {service.protocol for service in self.services}.intersection(self.SSL_PROTOCOLS):
            self.ssl_certfile = self.required("SSL_CERTFILE")
            self.ssl_keyfile = self.required("SSL_KEYFILE")
        self.report_services = self.services_to_report()

        # debug
        self.debug_skip_await_mempool_sync_on_startup = self.boolean("DEBUG_SKIP_AWAIT_MEMPOOL_SYNC_ON_STARTUP", False)

    def sane_max_sessions(self):
        """Return the maximum number of sessions to permit.  Normally this
        is MAX_SESSIONS.  However, to prevent open file exhaustion, adjust
        downwards if running with a small open file rlimit."""
        env_value = self.integer("MAX_SESSIONS", 1000)
        # No resource module on Windows
        try:
            import resource

            nofile_limit = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
            # We give the DB 250 files; allow ElectrumX 100 for itself
            value = max(0, min(env_value, nofile_limit - 350))
            if value < env_value:
                self.logger.warning(
                    f"lowered maximum sessions from {env_value:,d} to "
                    f"{value:,d} because your open file limit is "
                    f"{nofile_limit:,d}"
                )
        except ImportError:
            value = 512  # that is what returned by stdio's _getmaxstdio()
        return value

    def _check_and_fix_cost_limits(self):
        if self.cost_hard_limit < self.cost_soft_limit:
            raise self.Error(
                f"COST_HARD_LIMIT must be >= COST_SOFT_LIMIT. "
                f"got (COST_HARD_LIMIT={self.cost_hard_limit} "
                f"and COST_SOFT_LIMIT={self.cost_soft_limit})"
            )
        # hard limit should be strictly higher than soft limit (unless both are 0)
        if self.cost_hard_limit == self.cost_soft_limit and self.cost_soft_limit > 0:
            self.logger.info("found COST_HARD_LIMIT == COST_SOFT_LIMIT. " "bumping COST_HARD_LIMIT by 1.")
            self.cost_hard_limit = self.cost_soft_limit + 1

    def _parse_services(self, services_str, default_func):
        result = []
        for service_str in services_str.split(","):
            if not service_str:
                continue
            try:
                service = Service.from_string(service_str, default_func=default_func)
            except Exception as e:
                raise ServiceError(f'"{service_str}" invalid: {e}') from None
            if service.protocol not in self.KNOWN_PROTOCOLS:
                raise ServiceError(f'"{service_str}" invalid: unknown protocol')
            result.append(service)

        # Find duplicate addresses
        service_map = {service.address: [] for service in result}
        for service in result:
            service_map[service.address].append(service)
        for address, services in service_map.items():
            if len(services) > 1:
                raise ServiceError(f"address {address} has multiple services")

        return result

    def services_to_run(self):
        def default_part(protocol, part):
            return default_services.get(protocol, {}).get(part)

        default_services = {protocol: {ServicePart.HOST: "all_interfaces"} for protocol in self.KNOWN_PROTOCOLS}
        default_services["rpc"] = {
            ServicePart.HOST: "localhost",
            ServicePart.PORT: 8000,
        }
        services = self._parse_services(self.default("SERVICES", ""), default_part)

        # Find onion hosts
        for service in services:
            if str(service.host).endswith(".onion"):
                raise ServiceError(f"bad host for SERVICES: {service}")

        return services

    def services_to_report(self):
        services = self._parse_services(self.default("REPORT_SERVICES", ""), None)

        for service in services:
            if service.protocol == "rpc":
                raise ServiceError(f"bad protocol for REPORT_SERVICES: {service.protocol}")
            if isinstance(service.host, (IPv4Address, IPv6Address)):
                ip_addr = service.host
                if ip_addr.is_multicast or ip_addr.is_unspecified or (ip_addr.is_private and self.peer_announce):
                    raise ServiceError(f"bad IP address for REPORT_SERVICES: {ip_addr}")
            elif service.host.lower() == "localhost":
                raise ServiceError(f"bad host for REPORT_SERVICES: {service.host}")

        return services

    def set_gaze_network_report_config(self):
        gaze_network_report_enabled = None
        gazenw_report_enable_raw: str | None = self.default("GAZE_NETWORK_REPORTING_ENABLED", None)

        if gazenw_report_enable_raw is not None:
            gazenw_report_enable_raw = gazenw_report_enable_raw.strip().lower()
            raw_is_true = gazenw_report_enable_raw in ["1", "y", "yes", "t", "true"]
            raw_is_false = gazenw_report_enable_raw in ["0", "n", "no", "f", "false"]

            if raw_is_true and not raw_is_false:
                gaze_network_report_enabled = True
            if raw_is_false and not raw_is_true:
                gaze_network_report_enabled = False

        # set default value
        if gaze_network_report_enabled is None:
            gaze_network_report_enabled = False if is_test_environment() else True

        if not gaze_network_report_enabled:
            return None

        gaze_network_report_url = self.default("GAZE_NETWORK_REPORTING_URL", "https://indexer.api.gaze.network")
        gaze_network_report_name = self.required("GAZE_NETWORK_REPORTING_NAME")
        gaze_network_report_website_url = self.default("GAZE_NETWORK_REPORTING_WEBSITE_URL", None)
        gaze_network_report_indexer_api_url = self.default("GAZE_NETWORK_REPORTING_INDEXER_API_URL", None)

        return GazeNetworkReportConfig(
            gaze_network_report_url,
            gaze_network_report_name,
            gaze_network_report_website_url,
            gaze_network_report_indexer_api_url,
        )

    def peer_discovery_enum(self):
        pd = self.default("PEER_DISCOVERY", "on").strip().lower()
        if pd in ("off", ""):
            return self.PD_OFF
        elif pd == "self":
            return self.PD_SELF
        else:
            return self.PD_ON


@dataclass
class GazeNetworkReportConfig:
    url: str
    name: str
    website_url: "str | None"
    indexer_api_url: "str | None"
