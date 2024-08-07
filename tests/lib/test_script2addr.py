import pytest

from electrumx.lib.script2addr import (
    get_address_from_output_script,
    get_script_from_address,
)


def test_get_address_from_output_script():
    # the inverse of this test is in test_bitcoin: test_address_to_script
    def addr_from_script(script):
        return get_address_from_output_script(bytes.fromhex(script))

    # bech32/bech32m native segwit
    # test vectors from BIP-0173/BIP-0350
    assert "bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4" == addr_from_script(
        "0014751e76e8199196d454941c45d1b3a323f1433bd6"
    )
    assert "bc1pw508d6qejxtdg4y5r3zarvary0c5xw7kw508d6qejxtdg4y5r3zarvary0c5xw7kt5nd6y" == addr_from_script(
        "5128751e76e8199196d454941c45d1b3a323f1433bd6751e76e8199196d454941c45d1b3a323f1433bd6"
    )
    assert "bc1sw50qgdz25j" == addr_from_script("6002751e")
    assert "bc1zw508d6qejxtdg4y5r3zarvaryvaxxpcs" == addr_from_script("5210751e76e8199196d454941c45d1b3a323")
    assert "bc1p0xlxvlhemja6c4dqv22uapctqupfhlxm9h8z3k2e72q4k9hcz7vqzk5jj0" == addr_from_script(
        "512079be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798"
    )
    # almost but not quite
    assert None == addr_from_script("0013751e76e8199196d454941c45d1b3a323f1433b")

    # base58 p2pkh
    assert "14gcRovpkCoGkCNBivQBvw7eso7eiNAbxG" == addr_from_script(
        "76a91428662c67561b95c79d2257d2a93d9d151c977e9188ac"
    )
    assert "1BEqfzh4Y3zzLosfGhw1AsqbEKVW6e1qHv" == addr_from_script(
        "76a914704f4b81cadb7bf7e68c08cd3657220f680f863c88ac"
    )
    # almost but not quite
    assert None == addr_from_script("76a9130000000000000000000000000000000000000088ac")

    # base58 p2sh
    assert "35ZqQJcBQMZ1rsv8aSuJ2wkC7ohUCQMJbT" == addr_from_script("a9142a84cf00d47f699ee7bbc1dea5ec1bdecb4ac15487")
    assert "3PyjzJ3im7f7bcV724GR57edKDqoZvH7Ji" == addr_from_script("a914f47c8954e421031ad04ecd8e7752c9479206b9d387")
    # almost but not quite
    assert None == addr_from_script("a912f47c8954e421031ad04ecd8e7752c947920687")

    # p2pk
    assert None == addr_from_script("210289e14468d94537493c62e2168318b568912dec0fb95609afd56f2527c2751c8bac")
    assert None == addr_from_script(
        "41045485b0b076848af1209e788c893522a90f3df77c1abac2ca545846a725e6c3da1f7743f55a1bc3b5f0c7e0ee4459954ec0307022742d60032b13432953eb7120ac"
    )
    # almost but not quite
    assert None == addr_from_script("200289e14468d94537493c62e2168318b568912dec0fb95609afd56f2527c2751cac")
    assert None == addr_from_script("210589e14468d94537493c62e2168318b568912dec0fb95609afd56f2527c2751c8bac")


def test_get_script_from_address():
    def script_hex_from_addr(address: str) -> str:
        return get_script_from_address(address).hex()

    # bech32/bech32m native segwit
    # test vectors from BIP-0173/BIP-0350
    assert "0014751e76e8199196d454941c45d1b3a323f1433bd6" == script_hex_from_addr(
        "bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4"
    )
    assert (
        "5128751e76e8199196d454941c45d1b3a323f1433bd6751e76e8199196d454941c45d1b3a323f1433bd6"
        == script_hex_from_addr("bc1pw508d6qejxtdg4y5r3zarvary0c5xw7kw508d6qejxtdg4y5r3zarvary0c5xw7kt5nd6y")
    )
    assert "6002751e" == script_hex_from_addr("bc1sw50qgdz25j")
    assert "5210751e76e8199196d454941c45d1b3a323" == script_hex_from_addr("bc1zw508d6qejxtdg4y5r3zarvaryvaxxpcs")
    assert "512079be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798" == script_hex_from_addr(
        "bc1p0xlxvlhemja6c4dqv22uapctqupfhlxm9h8z3k2e72q4k9hcz7vqzk5jj0"
    )

    # base58 p2pkh
    assert "76a91428662c67561b95c79d2257d2a93d9d151c977e9188ac" == script_hex_from_addr(
        "14gcRovpkCoGkCNBivQBvw7eso7eiNAbxG"
    )
    assert "76a914704f4b81cadb7bf7e68c08cd3657220f680f863c88ac" == script_hex_from_addr(
        "1BEqfzh4Y3zzLosfGhw1AsqbEKVW6e1qHv"
    )

    # base58 p2sh
    assert "a9142a84cf00d47f699ee7bbc1dea5ec1bdecb4ac15487" == script_hex_from_addr(
        "35ZqQJcBQMZ1rsv8aSuJ2wkC7ohUCQMJbT"
    )
    assert "a914f47c8954e421031ad04ecd8e7752c9479206b9d387" == script_hex_from_addr(
        "3PyjzJ3im7f7bcV724GR57edKDqoZvH7Ji"
    )
