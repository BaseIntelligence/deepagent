import { Catalog } from "../src/catalog.js";

test("catalog add and get", () => {
  const c = new Catalog();
  c.add("a1", "alpha", ["x"]);
  expect(c.get("a1")).toEqual({ name: "alpha", tags: ["x"] });
});

test("catalog findByTag", () => {
  const c = new Catalog();
  c.add("a1", "alpha", ["x", "y"]);
  c.add("b1", "beta", ["y"]);
  const hits = c.findByTag("y");
  expect(hits).toHaveLength(2);
});
