import { Registry } from "../src/registry.js";

test("register assigns sequential ids", () => {
  const r = new Registry();
  const a = r.register("alpha", ["core"]);
  const b = r.register("beta");
  expect(a.id).toBe("svc-1");
  expect(b.id).toBe("svc-2");
  expect(r.count()).toBe(2);
});

test("lookup missing returns null", () => {
  const r = new Registry();
  expect(r.lookup("nope")).toBeNull();
});
