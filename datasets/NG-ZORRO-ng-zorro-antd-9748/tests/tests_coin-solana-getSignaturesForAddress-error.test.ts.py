jest.mock("@solana/web3.js", () => {
  const actual = jest.requireActual("@solana/web3.js");

  class MockConnection {
    public getSignaturesForAddress = jest.fn();
    public getBalance = jest.fn();
    public getLatestBlockhash = jest.fn();
    public getFeeForMessage = jest.fn();
    public getBalanceAndContext = jest.fn();
    public getParsedTokenAccountsByOwner = jest.fn();
    public getParsedProgramAccounts = jest.fn();
    public getInflationReward = jest.fn();
    public getVoteAccounts = jest.fn();
    public getParsedTransactions = jest.fn();
    public getParsedAccountInfo = jest.fn();
    public getMultipleParsedAccounts = jest.fn();
    public sendRawTransaction = jest.fn();
    public confirmTransaction = jest.fn();
    public getRecentPrioritizationFees = jest.fn();
    public simulateTransaction = jest.fn();

    constructor(..._args: unknown[]) {}
  }

  return {
    ...actual,
    Connection: MockConnection,
  };
});

import { NetworkError } from "@ledgerhq/errors";
import { getChainAPI } from "../libs/coin-modules/coin-solana/src/network/chain";

describe("coin-solana getSignaturesForAddress", () => {
  it("retries and remaps RPC -32020 long-term storage errors", async () => {
    const api = getChainAPI({ endpoint: "http://localhost:8899" });

    const rpcError = new Error(
      '{"code":-32020,"message":"Transaction history is not available from this node"}',
    );

    const mockedConnection = api.connection as unknown as {
      getSignaturesForAddress: jest.Mock;
    };

    mockedConnection.getSignaturesForAddress
      .mockRejectedValueOnce(rpcError)
      .mockRejectedValueOnce(rpcError)
      .mockResolvedValueOnce([{ signature: "sig1", slot: 123, blockTime: 1700000000, err: null }]);

    await expect(
      api.getSignaturesForAddress("HxCvgjSbF8HMt3fj8P3j49jmajNCMwKAqBu79HUDPtkM"),
    ).resolves.toEqual([{ signature: "sig1", slot: 123, blockTime: 1700000000, err: null }]);

    expect(mockedConnection.getSignaturesForAddress).toHaveBeenCalledTimes(3);
  });

  it("still remaps unrelated errors to NetworkError without retrying", async () => {
    const api = getChainAPI({ endpoint: "http://localhost:8899" });

    const rpcError = new Error("some other RPC failure");
    const mockedConnection = api.connection as unknown as {
      getSignaturesForAddress: jest.Mock;
    };

    mockedConnection.getSignaturesForAddress.mockRejectedValueOnce(rpcError);

    await expect(
      api.getSignaturesForAddress("HxCvgjSbF8HMt3fj8P3j49jmajNCMwKAqBu79HUDPtkM"),
    ).rejects.toEqual(new NetworkError("some other RPC failure"));

    expect(mockedConnection.getSignaturesForAddress).toHaveBeenCalledTimes(1);
  });
});
