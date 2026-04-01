import { describe, expect, it } from "vitest";
import { SqliteVfsPool } from "../rivetkit-typescript/packages/sqlite-vfs/src/pool";
import type { KvVfsOptions } from "@rivetkit/sqlite-vfs";

function keyToString(key: Uint8Array): string {
	return Buffer.from(key).toString("hex");
}

function createKvStore(): KvVfsOptions {
	const store = new Map<string, Uint8Array>();

	return {
		get: async (key) => {
			const value = store.get(keyToString(key));
			return value ? new Uint8Array(value) : null;
		},
		getBatch: async (keys) => {
			return keys.map((key) => {
				const value = store.get(keyToString(key));
				return value ? new Uint8Array(value) : null;
			});
		},
		put: async (key, value) => {
			store.set(keyToString(key), new Uint8Array(value));
		},
		putBatch: async (entries) => {
			for (const [key, value] of entries) {
				store.set(keyToString(key), new Uint8Array(value));
			}
		},
		deleteBatch: async (keys) => {
			for (const key of keys) {
				store.delete(keyToString(key));
			}
		},
	};
}

describe("SqliteVfsPool destroy", () => {
	it("destroying one idle instance does not break another instance's database", async () => {
		const pool = new SqliteVfsPool({
			actorsPerInstance: 1,
			idleDestroyMs: 20,
		});

		const handleA = await pool.acquire("actor-a");
		const handleB = await pool.acquire("actor-b");

		const dbA = await handleA.open("db-a", createKvStore());
		await dbA.exec("CREATE TABLE t (value TEXT)");
		await dbA.exec("INSERT INTO t (value) VALUES ('a')");
		await dbA.close();

		const dbB = await handleB.open("db-b", createKvStore());
		await dbB.exec("CREATE TABLE t (value TEXT)");
		await dbB.exec("INSERT INTO t (value) VALUES ('b')");

		await handleA.destroy();
		await new Promise((resolve) => setTimeout(resolve, 80));

		await expect(dbB.query("SELECT value FROM t")).resolves.toMatchObject({
			rows: [["b"]],
		});

		await dbB.close();
		await handleB.destroy();
		await pool.shutdown();
	});
});
