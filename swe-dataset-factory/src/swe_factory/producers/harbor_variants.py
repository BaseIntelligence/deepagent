"""Diversified Harbor motor inputs (≥10 hard multi-file packs).

Builds alternate multi-module FaultPlans against the same offline fixtures as
``harbor_motors`` so the ship pipeline can emit 10–15 DeepSWE packs with
Python / Go / TypeScript coverage without relaxing the multi-file floor.
"""

from __future__ import annotations

from dataclasses import replace

from swe_factory.producers.harbor_motors import (
    GO_KV_FAULT,
    GO_KV_HELD_OUT,
    MOTOR_SEEDS,
    PYTHON_ORDERS_FAULT,
    PYTHON_ORDERS_HELD_OUT,
    TS_REGISTRY_FAULT,
    TS_REGISTRY_HELD_OUT,
    FaultPlan,
    HarborMotorSeed,
    HeldOutTest,
)

# ---------------------------------------------------------------------------
# Python multi-module alternative faults (same green tree; distinct breaks)
# ---------------------------------------------------------------------------

PYTHON_FAULT_B = FaultPlan(
    description=(
        "Tax path becomes a no-op, reserve mutates stock without atomic rollback, "
        "and checkout total ignores the pricing engine cart path."
    ),
    replacements=(
        (
            "orderlib/pricing.py",
            "def apply_tax(self, amount: float) -> float:\n"
            "        return round(float(amount) * (1.0 + self.tax_rate), 2)",
            "def apply_tax(self, amount: float) -> float:\n"
            "        # Broken: tax never applied\n"
            "        return round(float(amount), 2)",
        ),
        (
            "orderlib/inventory.py",
            "def reserve(self, requests: dict[str, int]) -> bool:\n"
            '        """Atomically reserve all requested SKUs or none."""\n'
            "        if not self.can_reserve(requests):\n"
            "            return False\n"
            "        for sku, qty in requests.items():\n"
            "            self._stock[sku] = self.available(sku) - int(qty)\n"
            "        return True",
            "def reserve(self, requests: dict[str, int]) -> bool:\n"
            '        """Broken: partial reserves allowed; ignores can_reserve."""\n'
            "        for sku, qty in requests.items():\n"
            "            if self.available(sku) >= int(qty):\n"
            "                self._stock[sku] = self.available(sku) - int(qty)\n"
            "        return True",
        ),
        (
            "orderlib/checkout.py",
            "lines = [(item.unit_price, item.quantity) for item in items]\n"
            "        total = self.pricing.cart_total(lines)",
            "lines = [(item.unit_price, item.quantity) for item in items]\n"
            "        # Broken: raw sum skips discount+tax composition\n"
            "        total = sum(p * q for p, q in lines)",
        ),
    ),
    uses_not_implemented=False,
)

PYTHON_FAULT_C = FaultPlan(
    description=(
        "line_total mis-scales quantities, can_reserve flips inequality, and "
        "checkout fails to aggregate duplicate SKUs before reservation."
    ),
    replacements=(
        (
            "orderlib/pricing.py",
            "def line_total(self, unit_price: float, quantity: int) -> float:\n"
            "        if quantity < 0:\n"
            '            raise ValueError("quantity must be non-negative")\n'
            "        return float(unit_price) * int(quantity)",
            "def line_total(self, unit_price: float, quantity: int) -> float:\n"
            "        if quantity < 0:\n"
            '            raise ValueError("quantity must be non-negative")\n'
            "        # Broken: divides instead of multiplies\n"
            "        q = int(quantity) or 1\n"
            "        return float(unit_price) / q",
        ),
        (
            "orderlib/inventory.py",
            "def can_reserve(self, requests: dict[str, int]) -> bool:\n"
            "        for sku, qty in requests.items():\n"
            "            if qty < 0:\n"
            "                return False\n"
            "            if self.available(sku) < qty:\n"
            "                return False\n"
            "        return True",
            "def can_reserve(self, requests: dict[str, int]) -> bool:\n"
            "        for sku, qty in requests.items():\n"
            "            if qty < 0:\n"
            "                return False\n"
            "            # Broken: inverted availability check\n"
            "            if self.available(sku) > qty:\n"
            "                return False\n"
            "        return True",
        ),
        (
            "orderlib/checkout.py",
            "requests = {item.sku: item.quantity for item in items}\n"
            "        # Aggregate duplicate SKUs\n"
            "        aggregated: dict[str, int] = {}\n"
            "        for sku, qty in requests.items():\n"
            "            aggregated[sku] = aggregated.get(sku, 0) + int(qty)",
            "requests = {item.sku: item.quantity for item in items}\n"
            "        # Broken: no aggregation — last SKU wins on duplicates\n"
            "        aggregated: dict[str, int] = dict(requests)",
        ),
    ),
    uses_not_implemented=False,
)

PYTHON_FAULT_D = FaultPlan(
    description=(
        "Discount always fires regardless of threshold, release subtracts stock, "
        "and checkout marks empty carts as accepted."
    ),
    replacements=(
        (
            "orderlib/pricing.py",
            "def apply_discount(self, subtotal: float) -> float:\n"
            '        """10% off when subtotal reaches the configured threshold; else identity."""\n'
            "        sub = float(subtotal)\n"
            "        if sub >= self.discount_threshold:\n"
            "            return round(sub * 0.9, 2)\n"
            "        return round(sub, 2)",
            "def apply_discount(self, subtotal: float) -> float:\n"
            '        """Broken: always discounts 10% irrespective of threshold."""\n'
            "        sub = float(subtotal)\n"
            "        return round(sub * 0.9, 2)",
        ),
        (
            "orderlib/inventory.py",
            "def release(self, requests: dict[str, int]) -> None:\n"
            "        for sku, qty in requests.items():\n"
            "            if qty < 0:\n"
            '                raise ValueError("release quantity must be non-negative")\n'
            "            self._stock[sku] = self.available(sku) + int(qty)",
            "def release(self, requests: dict[str, int]) -> None:\n"
            "        for sku, qty in requests.items():\n"
            "            if qty < 0:\n"
            '                raise ValueError("release quantity must be non-negative")\n'
            "            # Broken: releases by subtracting (drains stock further)\n"
            "            self._stock[sku] = self.available(sku) - int(qty)",
        ),
        (
            "orderlib/checkout.py",
            "if not items:\n"
            "            return OrderResult(\n"
            "                accepted=False,\n"
            "                total=0.0,\n"
            "                remaining=self.inventory.snapshot(),\n"
            '                reason="empty_cart",\n'
            "            )",
            "if not items:\n"
            "            # Broken: empty cart incorrectly accepted\n"
            "            return OrderResult(\n"
            "                accepted=True,\n"
            "                total=0.0,\n"
            "                remaining=self.inventory.snapshot(),\n"
            '                reason="ok",\n'
            "            )",
        ),
    ),
    uses_not_implemented=False,
)

# ---------------------------------------------------------------------------
# Go multi-module alternative faults
# ---------------------------------------------------------------------------

GO_FAULT_B = FaultPlan(
    description=(
        "Store Set becomes a no-op for new keys gated incorrectly; Delete always "
        "reports true; Router Fetch returns empty without error."
    ),
    replacements=(
        (
            "store.go",
            "func (s *Store) Set(key, value string) {\n"
            "\tif s.data == nil {\n"
            "\t\ts.data = make(map[string]string)\n"
            "\t}\n"
            "\ts.data[key] = value\n"
            "}",
            "func (s *Store) Set(key, value string) {\n"
            "\tif s.data == nil {\n"
            "\t\ts.data = make(map[string]string)\n"
            "\t}\n"
            "\t// Broken: only overwrites existing keys; never inserts new ones\n"
            "\tif _, ok := s.data[key]; ok {\n"
            "\t\ts.data[key] = value\n"
            "\t}\n"
            "}",
        ),
        (
            "store.go",
            "func (s *Store) Delete(key string) bool {\n"
            "\tif s.data == nil {\n"
            "\t\treturn false\n"
            "\t}\n"
            "\tif _, ok := s.data[key]; !ok {\n"
            "\t\treturn false\n"
            "\t}\n"
            "\tdelete(s.data, key)\n"
            "\treturn true\n"
            "}",
            "func (s *Store) Delete(key string) bool {\n"
            "\tif s.data == nil {\n"
            "\t\treturn true\n"
            "\t}\n"
            "\t// Broken: always claims success even when missing\n"
            "\tdelete(s.data, key)\n"
            "\treturn true\n"
            "}",
        ),
        (
            "router.go",
            "func (r *Router) Fetch(key string) (string, error) {\n"
            "\tv, ok := r.store.Get(key)\n"
            "\tif !ok {\n"
            '\t\treturn "", fmt.Errorf("missing key %q", key)\n'
            "\t}\n"
            "\treturn v, nil\n"
            "}",
            "func (r *Router) Fetch(key string) (string, error) {\n"
            "\t// Broken: ignores existence and never errors\n"
            "\tv, _ := r.store.Get(key)\n"
            "\treturn v, nil\n"
            "}",
        ),
    ),
    uses_not_implemented=False,
)

GO_FAULT_C = FaultPlan(
    description=(
        "Size is off-by-one, Put accepts empty keys, and Remove always errors "
        "even when the key exists."
    ),
    replacements=(
        (
            "store.go",
            "func (s *Store) Size() int {\n\treturn len(s.data)\n}",
            "func (s *Store) Size() int {\n\t// Broken: off-by-one\n\treturn len(s.data) + 1\n}",
        ),
        (
            "router.go",
            "func (r *Router) Put(key, value string) error {\n"
            '\tif key == "" {\n'
            '\t\treturn fmt.Errorf("empty key")\n'
            "\t}\n"
            "\tr.store.Set(key, value)\n"
            "\treturn nil\n"
            "}",
            "func (r *Router) Put(key, value string) error {\n"
            "\t// Broken: empty keys accepted\n"
            "\tr.store.Set(key, value)\n"
            "\treturn nil\n"
            "}",
        ),
        (
            "router.go",
            "func (r *Router) Remove(key string) error {\n"
            "\tif !r.store.Delete(key) {\n"
            '\t\treturn fmt.Errorf("missing key %q", key)\n'
            "\t}\n"
            "\treturn nil\n"
            "}",
            "func (r *Router) Remove(key string) error {\n"
            "\t// Broken: always error after delete attempt\n"
            "\t_ = r.store.Delete(key)\n"
            '\treturn fmt.Errorf("missing key %q", key)\n'
            "}",
        ),
    ),
    uses_not_implemented=False,
)

GO_FAULT_D = FaultPlan(
    description=(
        "Get swaps value/ok semantics for missing keys, Upsert never writes, "
        "and Delete becomes a no-op."
    ),
    replacements=(
        (
            "store.go",
            "func (s *Store) Get(key string) (string, bool) {\n"
            "\tif s.data == nil {\n"
            '\t\treturn "", false\n'
            "\t}\n"
            "\tv, ok := s.data[key]\n"
            "\treturn v, ok\n"
            "}",
            "func (s *Store) Get(key string) (string, bool) {\n"
            "\tif s.data == nil {\n"
            '\t\treturn "ghost", true\n'
            "\t}\n"
            "\tv, ok := s.data[key]\n"
            "\tif !ok {\n"
            '\t\treturn "ghost", true\n'
            "\t}\n"
            "\treturn v, ok\n"
            "}",
        ),
        (
            "store.go",
            "func (s *Store) Delete(key string) bool {\n"
            "\tif s.data == nil {\n"
            "\t\treturn false\n"
            "\t}\n"
            "\tif _, ok := s.data[key]; !ok {\n"
            "\t\treturn false\n"
            "\t}\n"
            "\tdelete(s.data, key)\n"
            "\treturn true\n"
            "}",
            "func (s *Store) Delete(key string) bool {\n"
            "\t// Broken: never deletes\n"
            "\tif s.data == nil {\n"
            "\t\treturn false\n"
            "\t}\n"
            "\t_, ok := s.data[key]\n"
            "\treturn ok\n"
            "}",
        ),
        (
            "router.go",
            "func (r *Router) Upsert(key, value string) (int, error) {\n"
            "\tif err := r.Put(key, value); err != nil {\n"
            "\t\treturn 0, err\n"
            "\t}\n"
            "\treturn r.store.Size(), nil\n"
            "}",
            "func (r *Router) Upsert(key, value string) (int, error) {\n"
            "\t// Broken: never writes; returns zero size\n"
            '\tif key == "" {\n'
            '\t\treturn 0, fmt.Errorf("empty key")\n'
            "\t}\n"
            "\treturn 0, nil\n"
            "}",
        ),
    ),
    uses_not_implemented=False,
)

# ---------------------------------------------------------------------------
# TypeScript multi-module alternative faults
# ---------------------------------------------------------------------------

TS_FAULT_B = FaultPlan(
    description=(
        "Catalog get returns bare name only, register mutates caller tags and "
        "advances seq twice, listByTag drops non-matching incorrectly? full id "
        "path regression across modules."
    ),
    replacements=(
        (
            "src/catalog.js",
            "get(id) {\n    return this.items.get(id) ?? null;\n  }",
            "get(id) {\n"
            "    const hit = this.items.get(id);\n"
            "    // Broken: drops tags and returns only name\n"
            "    return hit ? { name: hit.name } : null;\n"
            "  }",
        ),
        (
            "src/catalog.js",
            "add(id, name, tags = []) {\n"
            "    if (!id || !name) {\n"
            '      throw new Error("id and name required");\n'
            "    }\n"
            "    this.items.set(id, { name, tags: [...tags] });\n"
            "  }",
            "add(id, name, tags = []) {\n"
            "    if (!id || !name) {\n"
            '      throw new Error("id and name required");\n'
            "    }\n"
            "    // Broken: stores alias reference, no copy\n"
            "    this.items.set(id, { name, tags });\n"
            "  }",
        ),
        (
            "src/registry.js",
            "register(name, tags = []) {\n"
            "    if (!name || !String(name).trim()) {\n"
            '      throw new Error("name required");\n'
            "    }\n"
            "    this._seq += 1;\n"
            "    const id = `svc-${this._seq}`;\n"
            "    this.catalog.add(id, name, tags);\n"
            "    return { id, name, tags: [...tags] };\n"
            "  }",
            "register(name, tags = []) {\n"
            "    if (!name || !String(name).trim()) {\n"
            '      throw new Error("name required");\n'
            "    }\n"
            "    // Broken: double-advance consumes ids; mutates tags\n"
            "    this._seq += 2;\n"
            "    const id = `svc-${this._seq}`;\n"
            "    tags.push('__tainted');\n"
            "    this.catalog.add(id, name, tags);\n"
            "    return { id, name, tags };\n"
            "  }",
        ),
    ),
    uses_not_implemented=False,
)

TS_FAULT_C = FaultPlan(
    description=(
        "findByTag case-mismatch and incomplete results; lookup always null; "
        "count reports sequence instead of catalog size."
    ),
    replacements=(
        (
            "src/catalog.js",
            "findByTag(tag) {\n"
            "    const out = [];\n"
            "    for (const [id, meta] of this.items.entries()) {\n"
            "      if (meta.tags.includes(tag)) {\n"
            "        out.push({ id, ...meta });\n"
            "      }\n"
            "    }\n"
            "    return out;\n"
            "  }",
            "findByTag(tag) {\n"
            "    const out = [];\n"
            "    for (const [id, meta] of this.items.entries()) {\n"
            "      // Broken: uppercase-only match and drops tags from payload\n"
            "      if (meta.tags.includes(String(tag).toUpperCase())) {\n"
            "        out.push({ id, name: meta.name });\n"
            "      }\n"
            "    }\n"
            "    return out;\n"
            "  }",
        ),
        (
            "src/registry.js",
            "lookup(id) {\n"
            "    const hit = this.catalog.get(id);\n"
            "    if (!hit) {\n"
            "      return null;\n"
            "    }\n"
            "    return { id, ...hit };\n"
            "  }",
            "lookup(id) {\n    // Broken: always null\n    void id;\n    return null;\n  }",
        ),
        (
            "src/registry.js",
            "count() {\n    return this.catalog.size();\n  }",
            "count() {\n"
            "    // Broken: returns sequence counter, not catalog size\n"
            "    return this._seq;\n"
            "  }",
        ),
    ),
    uses_not_implemented=False,
)

TS_FAULT_D = FaultPlan(
    description=(
        "Catalog size undercounts, registry listByTag truncates, and register "
        "reuses name as id instead of sequential svc-N."
    ),
    replacements=(
        (
            "src/catalog.js",
            "size() {\n    return this.items.size;\n  }",
            "size() {\n"
            "    // Broken: undercounts by one when non-empty\n"
            "    const n = this.items.size;\n"
            "    return n > 0 ? n - 1 : 0;\n"
            "  }",
        ),
        (
            "src/catalog.js",
            "get(id) {\n    return this.items.get(id) ?? null;\n  }",
            "get(id) {\n"
            "    // Broken: only returns rows whose id starts with svc-\n"
            "    if (!String(id).startsWith('svc-')) {\n"
            "      return null;\n"
            "    }\n"
            "    return this.items.get(id) ?? null;\n"
            "  }",
        ),
        (
            "src/registry.js",
            "register(name, tags = []) {\n"
            "    if (!name || !String(name).trim()) {\n"
            '      throw new Error("name required");\n'
            "    }\n"
            "    this._seq += 1;\n"
            "    const id = `svc-${this._seq}`;\n"
            "    this.catalog.add(id, name, tags);\n"
            "    return { id, name, tags: [...tags] };\n"
            "  }",
            "register(name, tags = []) {\n"
            "    if (!name || !String(name).trim()) {\n"
            '      throw new Error("name required");\n'
            "    }\n"
            "    // Broken: uses name as id; never advances sequence\n"
            "    const id = String(name);\n"
            "    this.catalog.add(id, name, tags);\n"
            "    return { id, name, tags: [...tags] };\n"
            "  }",
        ),
    ),
    uses_not_implemented=False,
)

# ---------------------------------------------------------------------------
# M8 scale variants (E–L) → ≥32 multi-file hybrid seeds for ≥30 gate
# ---------------------------------------------------------------------------

PYTHON_FAULT_E = FaultPlan(
    description=(
        "cart_total skips tax; available invents stock for missing SKUs; place rejects all carts."
    ),
    replacements=(
        (
            "orderlib/pricing.py",
            "def cart_total(self, lines: list[tuple[float, int]]) -> float:\n"
            "        subtotal = sum(self.line_total(price, qty) for price, qty in lines)\n"
            "        discounted = self.apply_discount(subtotal)\n"
            "        return self.apply_tax(discounted)",
            "def cart_total(self, lines: list[tuple[float, int]]) -> float:\n"
            "        subtotal = sum(self.line_total(price, qty) for price, qty in lines)\n"
            "        # Broken: returns discounted only (tax omitted)\n"
            "        return self.apply_discount(subtotal)",
        ),
        (
            "orderlib/inventory.py",
            "def available(self, sku: str) -> int:\n        return int(self._stock.get(sku, 0))",
            "def available(self, sku: str) -> int:\n"
            "        # Broken: invents stock for missing SKUs\n"
            "        return int(self._stock.get(sku, 99))",
        ),
        (
            "orderlib/checkout.py",
            "reserved = self.inventory.reserve(aggregated)\n"
            "        if not reserved:\n"
            "            return OrderResult(\n"
            "                accepted=False,\n"
            "                total=0.0,\n"
            "                remaining=self.inventory.snapshot(),\n"
            '                reason="reserve_failed",\n'
            "            )\n"
            "        lines = [(item.unit_price, item.quantity) for item in items]\n"
            "        total = self.pricing.cart_total(lines)\n"
            "        return OrderResult(\n"
            "            accepted=True,\n"
            "            total=total,\n"
            "            remaining=self.inventory.snapshot(),\n"
            '            reason="ok",\n'
            "        )",
            "reserved = self.inventory.reserve(aggregated)\n"
            "        # Broken: always rejects after reserve attempt (rolls back if needed)\n"
            "        if reserved:\n"
            "            self.inventory.release(aggregated)\n"
            "        return OrderResult(\n"
            "            accepted=False,\n"
            "            total=0.0,\n"
            "            remaining=self.inventory.snapshot(),\n"
            '            reason="reserve_failed",\n'
            "        )",
        ),
    ),
    uses_not_implemented=False,
)

PYTHON_FAULT_F = FaultPlan(
    description="tax doubles rate; can_reserve ignores qty sign; place_order drops lines pricing.",
    replacements=(
        (
            "orderlib/pricing.py",
            "def apply_tax(self, amount: float) -> float:\n"
            "        return round(float(amount) * (1.0 + self.tax_rate), 2)",
            "def apply_tax(self, amount: float) -> float:\n"
            "        # Broken: applies tax_rate twice\n"
            "        return round(float(amount) * (1.0 + self.tax_rate * 2), 2)",
        ),
        (
            "orderlib/inventory.py",
            "def can_reserve(self, requests: dict[str, int]) -> bool:\n"
            "        for sku, qty in requests.items():\n"
            "            if qty < 0:\n"
            "                return False\n"
            "            if self.available(sku) < qty:\n"
            "                return False\n"
            "        return True",
            "def can_reserve(self, requests: dict[str, int]) -> bool:\n"
            "        for sku, qty in requests.items():\n"
            "            # Broken: never checks negative qty\n"
            "            if self.available(sku) < abs(int(qty)):\n"
            "                return False\n"
            "        return True",
        ),
        (
            "orderlib/checkout.py",
            "lines = [(item.unit_price, item.quantity) for item in items]\n"
            "        total = self.pricing.cart_total(lines)\n"
            "        return OrderResult(\n"
            "            accepted=True,\n"
            "            total=total,\n"
            "            remaining=self.inventory.snapshot(),\n"
            '            reason="ok",\n'
            "        )",
            "lines = [(item.unit_price, item.quantity) for item in items]\n"
            "        # Broken: weights first line only\n"
            "        total = self.pricing.cart_total(lines[:1] or [(0.0, 0)])\n"
            "        return OrderResult(\n"
            "            accepted=True,\n"
            "            total=total,\n"
            "            remaining=self.inventory.snapshot(),\n"
            '            reason="ok",\n'
            "        )",
        ),
    ),
    uses_not_implemented=False,
)

PYTHON_FAULT_G = FaultPlan(
    description="discount threshold uses <; release no-ops; checkout skips can_reserve.",
    replacements=(
        (
            "orderlib/pricing.py",
            "def apply_discount(self, subtotal: float) -> float:\n"
            '        """10% off when subtotal reaches the configured threshold; else identity."""\n'
            "        sub = float(subtotal)\n"
            "        if sub >= self.discount_threshold:\n"
            "            return round(sub * 0.9, 2)\n"
            "        return round(sub, 2)",
            "def apply_discount(self, subtotal: float) -> float:\n"
            '        """Broken: strict greater-than never hits exact threshold."""\n'
            "        sub = float(subtotal)\n"
            "        if sub > self.discount_threshold:\n"
            "            return round(sub * 0.9, 2)\n"
            "        return round(sub, 2)",
        ),
        (
            "orderlib/inventory.py",
            "def release(self, requests: dict[str, int]) -> None:\n"
            "        for sku, qty in requests.items():\n"
            "            if qty < 0:\n"
            '                raise ValueError("release quantity must be non-negative")\n'
            "            self._stock[sku] = self.available(sku) + int(qty)",
            "def release(self, requests: dict[str, int]) -> None:\n"
            "        # Broken: release is a pure no-op\n"
            "        for sku, qty in requests.items():\n"
            "            if qty < 0:\n"
            '                raise ValueError("release quantity must be non-negative")\n'
            "            _ = (sku, qty)\n"
            "        return",
        ),
        (
            "orderlib/checkout.py",
            "if not self.inventory.can_reserve(aggregated):\n"
            "            return OrderResult(\n"
            "                accepted=False,\n"
            "                total=0.0,\n"
            "                remaining=self.inventory.snapshot(),\n"
            '                reason="insufficient_stock",\n'
            "            )",
            "# Broken: skips can_reserve gate entirely\n"
            "        if False and not self.inventory.can_reserve(aggregated):\n"
            "            return OrderResult(\n"
            "                accepted=False,\n"
            "                total=0.0,\n"
            "                remaining=self.inventory.snapshot(),\n"
            '                reason="insufficient_stock",\n'
            "            )",
        ),
    ),
    uses_not_implemented=False,
)

PYTHON_FAULT_H = FaultPlan(
    description=(
        "line_total ignores quantity; reserve doubles qty; checkout reason always empty cart."
    ),
    replacements=(
        (
            "orderlib/pricing.py",
            "def line_total(self, unit_price: float, quantity: int) -> float:\n"
            "        if quantity < 0:\n"
            '            raise ValueError("quantity must be non-negative")\n'
            "        return float(unit_price) * int(quantity)",
            "def line_total(self, unit_price: float, quantity: int) -> float:\n"
            "        if quantity < 0:\n"
            '            raise ValueError("quantity must be non-negative")\n'
            "        # Broken: always unit price of one unit\n"
            "        return float(unit_price)",
        ),
        (
            "orderlib/inventory.py",
            "def reserve(self, requests: dict[str, int]) -> bool:\n"
            '        """Atomically reserve all requested SKUs or none."""\n'
            "        if not self.can_reserve(requests):\n"
            "            return False\n"
            "        for sku, qty in requests.items():\n"
            "            self._stock[sku] = self.available(sku) - int(qty)\n"
            "        return True",
            "def reserve(self, requests: dict[str, int]) -> bool:\n"
            '        """Broken: doubles quantities when reserving."""\n'
            "        if not self.can_reserve(requests):\n"
            "            return False\n"
            "        for sku, qty in requests.items():\n"
            "            self._stock[sku] = self.available(sku) - int(qty) * 2\n"
            "        return True",
        ),
        (
            "orderlib/checkout.py",
            "if not items:\n"
            "            return OrderResult(\n"
            "                accepted=False,\n"
            "                total=0.0,\n"
            "                remaining=self.inventory.snapshot(),\n"
            '                reason="empty_cart",\n'
            "            )",
            "if not items:\n"
            "            # Broken: wrong reason string for empty cart\n"
            "            return OrderResult(\n"
            "                accepted=False,\n"
            "                total=0.0,\n"
            "                remaining=self.inventory.snapshot(),\n"
            '                reason="ok",\n'
            "            )",
        ),
    ),
    uses_not_implemented=False,
)

PYTHON_FAULT_I = FaultPlan(
    description="cart_total multiplies tax only; snapshot returns empty; place inverts acceptance.",
    replacements=(
        (
            "orderlib/pricing.py",
            "def cart_total(self, lines: list[tuple[float, int]]) -> float:\n"
            "        subtotal = sum(self.line_total(price, qty) for price, qty in lines)\n"
            "        discounted = self.apply_discount(subtotal)\n"
            "        return self.apply_tax(discounted)",
            "def cart_total(self, lines: list[tuple[float, int]]) -> float:\n"
            "        # Broken: tax applied to raw sum without discount\n"
            "        subtotal = sum(self.line_total(price, qty) for price, qty in lines)\n"
            "        return self.apply_tax(subtotal)",
        ),
        (
            "orderlib/inventory.py",
            "def snapshot(self) -> dict[str, int]:\n        return deepcopy(self._stock)",
            "def snapshot(self) -> dict[str, int]:\n"
            "        # Broken: always empty snapshot\n"
            "        return {}",
        ),
        (
            "orderlib/checkout.py",
            "return OrderResult(\n"
            "            accepted=True,\n"
            "            total=total,\n"
            "            remaining=self.inventory.snapshot(),\n"
            '            reason="ok",\n'
            "        )",
            "return OrderResult(\n"
            "            # Broken: flips acceptance and reason\n"
            "            accepted=False,\n"
            "            total=total,\n"
            "            remaining=self.inventory.snapshot(),\n"
            '            reason="insufficient_stock",\n'
            "        )",
        ),
    ),
    uses_not_implemented=False,
)

PYTHON_FAULT_J = FaultPlan(
    description="discount 50%; available halves; aggregated overwrite drops first SKU copy.",
    replacements=(
        (
            "orderlib/pricing.py",
            "if sub >= self.discount_threshold:\n"
            "            return round(sub * 0.9, 2)\n"
            "        return round(sub, 2)",
            "if sub >= self.discount_threshold:\n"
            "            return round(sub * 0.5, 2)\n"
            "        return round(sub, 2)",
        ),
        (
            "orderlib/inventory.py",
            "def available(self, sku: str) -> int:\n        return int(self._stock.get(sku, 0))",
            "def available(self, sku: str) -> int:\n"
            "        # Broken: reports half stock (floor)\n"
            "        return int(self._stock.get(sku, 0)) // 2",
        ),
        (
            "orderlib/checkout.py",
            "requests = {item.sku: item.quantity for item in items}\n"
            "        # Aggregate duplicate SKUs\n"
            "        aggregated: dict[str, int] = {}\n"
            "        for sku, qty in requests.items():\n"
            "            aggregated[sku] = aggregated.get(sku, 0) + int(qty)",
            "requests = {item.sku: item.quantity for item in items}\n"
            "        # Broken: uses pre-collapsed dict only (loses multi LineItem dupes)\n"
            "        aggregated: dict[str, int] = {sku: int(qty) for sku, qty in requests.items()}",
        ),
    ),
    uses_not_implemented=False,
)

PYTHON_FAULT_K = FaultPlan(
    description="tax zeroed; can_reserve always True; reserve mutates unvalidated.",
    replacements=(
        (
            "orderlib/pricing.py",
            "def apply_tax(self, amount: float) -> float:\n"
            "        return round(float(amount) * (1.0 + self.tax_rate), 2)",
            "def apply_tax(self, amount: float) -> float:\n"
            "        # Broken: tax-free always\n"
            "        return round(float(amount), 2)",
        ),
        (
            "orderlib/inventory.py",
            "def can_reserve(self, requests: dict[str, int]) -> bool:\n"
            "        for sku, qty in requests.items():\n"
            "            if qty < 0:\n"
            "                return False\n"
            "            if self.available(sku) < qty:\n"
            "                return False\n"
            "        return True",
            "def can_reserve(self, requests: dict[str, int]) -> bool:\n"
            "        # Broken: always allows reserve\n"
            "        _ = requests\n"
            "        return True",
        ),
        (
            "orderlib/inventory.py",
            "def reserve(self, requests: dict[str, int]) -> bool:\n"
            '        """Atomically reserve all requested SKUs or none."""\n'
            "        if not self.can_reserve(requests):\n"
            "            return False\n"
            "        for sku, qty in requests.items():\n"
            "            self._stock[sku] = self.available(sku) - int(qty)\n"
            "        return True",
            "def reserve(self, requests: dict[str, int]) -> bool:\n"
            '        """Broken: subtracts without clamping, can go negative."""\n'
            "        for sku, qty in requests.items():\n"
            "            self._stock[sku] = self.available(sku) - int(qty)\n"
            "        return True",
        ),
    ),
    uses_not_implemented=False,
)

PYTHON_FAULT_L = FaultPlan(
    description="line_total +qty bump; release multiplies; checkout accepts insufficient path.",
    replacements=(
        (
            "orderlib/pricing.py",
            "return float(unit_price) * int(quantity)",
            "return float(unit_price) * (int(quantity) + 1)",
        ),
        (
            "orderlib/inventory.py",
            "def release(self, requests: dict[str, int]) -> None:\n"
            "        for sku, qty in requests.items():\n"
            "            if qty < 0:\n"
            '                raise ValueError("release quantity must be non-negative")\n'
            "            self._stock[sku] = self.available(sku) + int(qty)",
            "def release(self, requests: dict[str, int]) -> None:\n"
            "        for sku, qty in requests.items():\n"
            "            if qty < 0:\n"
            '                raise ValueError("release quantity must be non-negative")\n'
            "            # Broken: multiplies stock instead of adding qty\n"
            "            self._stock[sku] = self.available(sku) * max(int(qty), 1)",
        ),
        (
            "orderlib/checkout.py",
            '                reason="insufficient_stock",\n'
            "            )\n"
            "        reserved = self.inventory.reserve(aggregated)",
            '                reason="ok",\n'
            "            )\n"
            "        reserved = self.inventory.reserve(aggregated)",
        ),
    ),
    uses_not_implemented=False,
)

GO_FAULT_E = FaultPlan(
    description="Set ignores empty value writes; Fetch always ok string; Size doubles.",
    replacements=(
        (
            "store.go",
            "func (s *Store) Set(key, value string) {\n"
            "\tif s.data == nil {\n"
            "\t\ts.data = make(map[string]string)\n"
            "\t}\n"
            "\ts.data[key] = value\n"
            "}",
            "func (s *Store) Set(key, value string) {\n"
            "\tif s.data == nil {\n"
            "\t\ts.data = make(map[string]string)\n"
            "\t}\n"
            "\t// Broken: skips empty values\n"
            '\tif value == "" {\n'
            "\t\treturn\n"
            "\t}\n"
            "\ts.data[key] = value\n"
            "}",
        ),
        (
            "store.go",
            "func (s *Store) Size() int {\n\treturn len(s.data)\n}",
            "func (s *Store) Size() int {\n\t// Broken: double count\n\treturn len(s.data) * 2\n}",
        ),
        (
            "router.go",
            "func (r *Router) Fetch(key string) (string, error) {\n"
            "\tv, ok := r.store.Get(key)\n"
            "\tif !ok {\n"
            '\t\treturn "", fmt.Errorf("missing key %q", key)\n'
            "\t}\n"
            "\treturn v, nil\n"
            "}",
            "func (r *Router) Fetch(key string) (string, error) {\n"
            "\t// Broken: fabricates value on miss\n"
            "\tv, ok := r.store.Get(key)\n"
            "\tif !ok {\n"
            '\t\treturn "missing", nil\n'
            "\t}\n"
            "\treturn v, nil\n"
            "}",
        ),
    ),
    uses_not_implemented=False,
)

GO_FAULT_F = FaultPlan(
    description="Delete never true; Put writes reversed; Upsert size always 0.",
    replacements=(
        (
            "store.go",
            "func (s *Store) Delete(key string) bool {\n"
            "\tif s.data == nil {\n"
            "\t\treturn false\n"
            "\t}\n"
            "\tif _, ok := s.data[key]; !ok {\n"
            "\t\treturn false\n"
            "\t}\n"
            "\tdelete(s.data, key)\n"
            "\treturn true\n"
            "}",
            "func (s *Store) Delete(key string) bool {\n"
            "\t// Broken: deletes silently then falsifies result\n"
            "\tif s.data == nil {\n"
            "\t\treturn false\n"
            "\t}\n"
            "\tdelete(s.data, key)\n"
            "\treturn false\n"
            "}",
        ),
        (
            "router.go",
            "func (r *Router) Put(key, value string) error {\n"
            '\tif key == "" {\n'
            '\t\treturn fmt.Errorf("empty key")\n'
            "\t}\n"
            "\tr.store.Set(key, value)\n"
            "\treturn nil\n"
            "}",
            "func (r *Router) Put(key, value string) error {\n"
            '\tif key == "" {\n'
            '\t\treturn fmt.Errorf("empty key")\n'
            "\t}\n"
            "\t// Broken: stores reversed key/value\n"
            "\tr.store.Set(value, key)\n"
            "\treturn nil\n"
            "}",
        ),
        (
            "router.go",
            "func (r *Router) Upsert(key, value string) (int, error) {\n"
            "\tif err := r.Put(key, value); err != nil {\n"
            "\t\treturn 0, err\n"
            "\t}\n"
            "\treturn r.store.Size(), nil\n"
            "}",
            "func (r *Router) Upsert(key, value string) (int, error) {\n"
            "\tif err := r.Put(key, value); err != nil {\n"
            "\t\treturn 0, err\n"
            "\t}\n"
            "\t// Broken: always zero size\n"
            "\treturn 0, nil\n"
            "}",
        ),
    ),
    uses_not_implemented=False,
)

GO_FAULT_G = FaultPlan(
    description="Get returns value without ok; Remove always nil; Size constant 1.",
    replacements=(
        (
            "store.go",
            "func (s *Store) Get(key string) (string, bool) {\n"
            "\tif s.data == nil {\n"
            '\t\treturn "", false\n'
            "\t}\n"
            "\tv, ok := s.data[key]\n"
            "\treturn v, ok\n"
            "}",
            "func (s *Store) Get(key string) (string, bool) {\n"
            "\tif s.data == nil {\n"
            '\t\treturn "", false\n'
            "\t}\n"
            "\tv, _ := s.data[key]\n"
            "\t// Broken: ok follows non-empty value only\n"
            '\treturn v, v != ""\n'
            "}",
        ),
        (
            "store.go",
            "func (s *Store) Size() int {\n\treturn len(s.data)\n}",
            "func (s *Store) Size() int {\n\t// Broken: constant one\n\treturn 1\n}",
        ),
        (
            "router.go",
            "func (r *Router) Remove(key string) error {\n"
            "\tif !r.store.Delete(key) {\n"
            '\t\treturn fmt.Errorf("missing key %q", key)\n'
            "\t}\n"
            "\treturn nil\n"
            "}",
            "func (r *Router) Remove(key string) error {\n"
            "\t// Broken: always succeeds\n"
            "\t_ = r.store.Delete(key)\n"
            "\treturn nil\n"
            "}",
        ),
    ),
    uses_not_implemented=False,
)

GO_FAULT_H = FaultPlan(
    description="Set prefixes values; Put allows empty; Fetch uppercases value.",
    replacements=(
        (
            "store.go",
            "func (s *Store) Set(key, value string) {\n"
            "\tif s.data == nil {\n"
            "\t\ts.data = make(map[string]string)\n"
            "\t}\n"
            "\ts.data[key] = value\n"
            "}",
            "func (s *Store) Set(key, value string) {\n"
            "\tif s.data == nil {\n"
            "\t\ts.data = make(map[string]string)\n"
            "\t}\n"
            "\t// Broken: prefixes stored value\n"
            '\ts.data[key] = "x:" + value\n'
            "}",
        ),
        (
            "router.go",
            "func (r *Router) Put(key, value string) error {\n"
            '\tif key == "" {\n'
            '\t\treturn fmt.Errorf("empty key")\n'
            "\t}\n"
            "\tr.store.Set(key, value)\n"
            "\treturn nil\n"
            "}",
            "func (r *Router) Put(key, value string) error {\n"
            "\t// Broken: empty keys ok\n"
            "\tr.store.Set(key, value)\n"
            "\treturn nil\n"
            "}",
        ),
        (
            "router.go",
            "func (r *Router) Fetch(key string) (string, error) {\n"
            "\tv, ok := r.store.Get(key)\n"
            "\tif !ok {\n"
            '\t\treturn "", fmt.Errorf("missing key %q", key)\n'
            "\t}\n"
            "\treturn v, nil\n"
            "}",
            "func (r *Router) Fetch(key string) (string, error) {\n"
            "\tv, ok := r.store.Get(key)\n"
            "\tif !ok {\n"
            '\t\treturn "", fmt.Errorf("missing key %q", key)\n'
            "\t}\n"
            "\t// Broken: mutates returned value\n"
            "\treturn v + v, nil\n"
            "}",
        ),
    ),
    uses_not_implemented=False,
)

GO_FAULT_I = FaultPlan(
    description="Delete clears entire map; Upsert double puts; Size off-by under.",
    replacements=(
        (
            "store.go",
            "func (s *Store) Delete(key string) bool {\n"
            "\tif s.data == nil {\n"
            "\t\treturn false\n"
            "\t}\n"
            "\tif _, ok := s.data[key]; !ok {\n"
            "\t\treturn false\n"
            "\t}\n"
            "\tdelete(s.data, key)\n"
            "\treturn true\n"
            "}",
            "func (s *Store) Delete(key string) bool {\n"
            "\tif s.data == nil {\n"
            "\t\treturn false\n"
            "\t}\n"
            "\t// Broken: wipes the whole store\n"
            "\ts.data = make(map[string]string)\n"
            "\t_ = key\n"
            "\treturn true\n"
            "}",
        ),
        (
            "store.go",
            "func (s *Store) Size() int {\n\treturn len(s.data)\n}",
            "func (s *Store) Size() int {\n"
            "\tn := len(s.data)\n"
            "\tif n == 0 {\n"
            "\t\treturn 0\n"
            "\t}\n"
            "\t// Broken: undercount\n"
            "\treturn n - 1\n"
            "}",
        ),
        (
            "router.go",
            "func (r *Router) Upsert(key, value string) (int, error) {\n"
            "\tif err := r.Put(key, value); err != nil {\n"
            "\t\treturn 0, err\n"
            "\t}\n"
            "\treturn r.store.Size(), nil\n"
            "}",
            "func (r *Router) Upsert(key, value string) (int, error) {\n"
            "\t// Broken: double put with different value\n"
            "\tif err := r.Put(key, value); err != nil {\n"
            "\t\treturn 0, err\n"
            "\t}\n"
            '\t_ = r.Put(key, value+"!")\n'
            "\treturn r.store.Size(), nil\n"
            "}",
        ),
    ),
    uses_not_implemented=False,
)

GO_FAULT_J = FaultPlan(
    description="Get swap args importantly wrong; Remove deletes always errors on success path.",
    replacements=(
        (
            "store.go",
            "func (s *Store) Get(key string) (string, bool) {\n"
            "\tif s.data == nil {\n"
            '\t\treturn "", false\n'
            "\t}\n"
            "\tv, ok := s.data[key]\n"
            "\treturn v, ok\n"
            "}",
            "func (s *Store) Get(key string) (string, bool) {\n"
            "\tif s.data == nil {\n"
            '\t\treturn "", false\n'
            "\t}\n"
            "\tv, ok := s.data[key]\n"
            "\t// Broken: returns inverted ok\n"
            "\treturn v, !ok\n"
            "}",
        ),
        (
            "router.go",
            "func (r *Router) Remove(key string) error {\n"
            "\tif !r.store.Delete(key) {\n"
            '\t\treturn fmt.Errorf("missing key %q", key)\n'
            "\t}\n"
            "\treturn nil\n"
            "}",
            "func (r *Router) Remove(key string) error {\n"
            "\tok := r.store.Delete(key)\n"
            "\tif ok {\n"
            "\t\t// Broken: errors after successful delete\n"
            '\t\treturn fmt.Errorf("missing key %q", key)\n'
            "\t}\n"
            "\treturn nil\n"
            "}",
        ),
        (
            "router.go",
            "func (r *Router) Put(key, value string) error {\n"
            '\tif key == "" {\n'
            '\t\treturn fmt.Errorf("empty key")\n'
            "\t}\n"
            "\tr.store.Set(key, value)\n"
            "\treturn nil\n"
            "}",
            "func (r *Router) Put(key, value string) error {\n"
            '\tif key == "" {\n'
            '\t\treturn fmt.Errorf("empty key")\n'
            "\t}\n"
            "\t// Broken: drops value\n"
            '\tr.store.Set(key, "")\n'
            "\treturn nil\n"
            "}",
        ),
    ),
    uses_not_implemented=False,
)

GO_FAULT_K = FaultPlan(
    description="Set uses key as value; Size nil-safe wrong; Fetch errs always.",
    replacements=(
        (
            "store.go",
            "func (s *Store) Set(key, value string) {\n"
            "\tif s.data == nil {\n"
            "\t\ts.data = make(map[string]string)\n"
            "\t}\n"
            "\ts.data[key] = value\n"
            "}",
            "func (s *Store) Set(key, value string) {\n"
            "\tif s.data == nil {\n"
            "\t\ts.data = make(map[string]string)\n"
            "\t}\n"
            "\t// Broken: stores key as value\n"
            "\ts.data[key] = key\n"
            "\t_ = value\n"
            "}",
        ),
        (
            "store.go",
            "func (s *Store) Size() int {\n\treturn len(s.data)\n}",
            "func (s *Store) Size() int {\n"
            "\tif s.data == nil {\n"
            "\t\treturn -1\n"
            "\t}\n"
            "\treturn len(s.data)\n"
            "}",
        ),
        (
            "router.go",
            "func (r *Router) Fetch(key string) (string, error) {\n"
            "\tv, ok := r.store.Get(key)\n"
            "\tif !ok {\n"
            '\t\treturn "", fmt.Errorf("missing key %q", key)\n'
            "\t}\n"
            "\treturn v, nil\n"
            "}",
            "func (r *Router) Fetch(key string) (string, error) {\n"
            "\t// Broken: always err\n"
            "\tv, _ := r.store.Get(key)\n"
            '\treturn v, fmt.Errorf("missing key %q", key)\n'
            "}",
        ),
    ),
    uses_not_implemented=False,
)

GO_FAULT_L = FaultPlan(
    description="Delete ignorekey; Upsert uninit; Put double freent.",
    replacements=(
        (
            "store.go",
            "func (s *Store) Delete(key string) bool {\n"
            "\tif s.data == nil {\n"
            "\t\treturn false\n"
            "\t}\n"
            "\tif _, ok := s.data[key]; !ok {\n"
            "\t\treturn false\n"
            "\t}\n"
            "\tdelete(s.data, key)\n"
            "\treturn true\n"
            "}",
            "func (s *Store) Delete(key string) bool {\n"
            "\tif s.data == nil {\n"
            "\t\treturn false\n"
            "\t}\n"
            "\t// Broken: checks wrong key always\n"
            "\tif _, ok := s.data[key+key]; !ok {\n"
            "\t\treturn false\n"
            "\t}\n"
            "\tdelete(s.data, key)\n"
            "\treturn true\n"
            "}",
        ),
        (
            "router.go",
            "func (r *Router) Upsert(key, value string) (int, error) {\n"
            "\tif err := r.Put(key, value); err != nil {\n"
            "\t\treturn 0, err\n"
            "\t}\n"
            "\treturn r.store.Size(), nil\n"
            "}",
            "func (r *Router) Upsert(key, value string) (int, error) {\n"
            "\t// Broken: never puts; invents size\n"
            '\tif key == "" {\n'
            '\t\treturn 0, fmt.Errorf("empty key")\n'
            "\t}\n"
            "\t_ = value\n"
            "\treturn 99, nil\n"
            "}",
        ),
        (
            "router.go",
            "func (r *Router) Put(key, value string) error {\n"
            '\tif key == "" {\n'
            '\t\treturn fmt.Errorf("empty key")\n'
            "\t}\n"
            "\tr.store.Set(key, value)\n"
            "\treturn nil\n"
            "}",
            "func (r *Router) Put(key, value string) error {\n"
            '\tif key == "" {\n'
            '\t\treturn fmt.Errorf("empty key")\n'
            "\t}\n"
            "\tr.store.Set(key, value)\n"
            "\t// Broken: second write clears\n"
            '\tr.store.Set(key, "")\n'
            "\treturn nil\n"
            "}",
        ),
    ),
    uses_not_implemented=False,
)

TS_FAULT_E = FaultPlan(
    description=("add drops tags; register seq starts at 10; count uses seq."),
    replacements=(
        (
            "src/catalog.js",
            "add(id, name, tags = []) {\n"
            "    if (!id || !name) {\n"
            '      throw new Error("id and name required");\n'
            "    }\n"
            "    this.items.set(id, { name, tags: [...tags] });\n"
            "  }",
            "add(id, name, tags = []) {\n"
            "    if (!id || !name) {\n"
            '      throw new Error("id and name required");\n'
            "    }\n"
            "    // Broken: drops tags entirely\n"
            "    this.items.set(id, { name, tags: [] });\n"
            "  }",
        ),
        (
            "src/registry.js",
            "register(name, tags = []) {\n"
            "    if (!name || !String(name).trim()) {\n"
            '      throw new Error("name required");\n'
            "    }\n"
            "    this._seq += 1;\n"
            "    const id = `svc-${this._seq}`;\n"
            "    this.catalog.add(id, name, tags);\n"
            "    return { id, name, tags: [...tags] };\n"
            "  }",
            "register(name, tags = []) {\n"
            "    if (!name || !String(name).trim()) {\n"
            '      throw new Error("name required");\n'
            "    }\n"
            "    // Broken: starts seq at 10\n"
            "    if (this._seq < 10) this._seq = 10;\n"
            "    this._seq += 1;\n"
            "    const id = `svc-${this._seq}`;\n"
            "    this.catalog.add(id, name, tags);\n"
            "    return { id, name, tags: [...tags] };\n"
            "  }",
        ),
        (
            "src/registry.js",
            "count() {\n    return this.catalog.size();\n  }",
            "count() {\n    // Broken: returns seq\n    return this._seq;\n  }",
        ),
    ),
    uses_not_implemented=False,
)

TS_FAULT_F = FaultPlan(
    description="findByTag duplicates; lookup spreads no id; size always 0.",
    replacements=(
        (
            "src/catalog.js",
            "findByTag(tag) {\n"
            "    const out = [];\n"
            "    for (const [id, meta] of this.items.entries()) {\n"
            "      if (meta.tags.includes(tag)) {\n"
            "        out.push({ id, ...meta });\n"
            "      }\n"
            "    }\n"
            "    return out;\n"
            "  }",
            "findByTag(tag) {\n"
            "    const out = [];\n"
            "    for (const [id, meta] of this.items.entries()) {\n"
            "      if (meta.tags.includes(tag)) {\n"
            "        // Broken: push twice\n"
            "        out.push({ id, ...meta });\n"
            "        out.push({ id, ...meta });\n"
            "      }\n"
            "    }\n"
            "    return out;\n"
            "  }",
        ),
        (
            "src/catalog.js",
            "size() {\n    return this.items.size;\n  }",
            "size() {\n    // Broken: always 0\n    return 0;\n  }",
        ),
        (
            "src/registry.js",
            "lookup(id) {\n"
            "    const hit = this.catalog.get(id);\n"
            "    if (!hit) {\n"
            "      return null;\n"
            "    }\n"
            "    return { id, ...hit };\n"
            "  }",
            "lookup(id) {\n"
            "    const hit = this.catalog.get(id);\n"
            "    if (!hit) {\n"
            "      return null;\n"
            "    }\n"
            "    // Broken: omit id field\n"
            "    return { ...hit };\n"
            "  }",
        ),
    ),
    uses_not_implemented=False,
)

TS_FAULT_G = FaultPlan(
    description="get returns name only string; register truncates name; listByTag empty.",
    replacements=(
        (
            "src/catalog.js",
            "get(id) {\n    return this.items.get(id) ?? null;\n  }",
            "get(id) {\n"
            "    const hit = this.items.get(id);\n"
            "    // Broken: string name only\n"
            "    return hit ? hit.name : null;\n"
            "  }",
        ),
        (
            "src/registry.js",
            "register(name, tags = []) {\n"
            "    if (!name || !String(name).trim()) {\n"
            '      throw new Error("name required");\n'
            "    }\n"
            "    this._seq += 1;\n"
            "    const id = `svc-${this._seq}`;\n"
            "    this.catalog.add(id, name, tags);\n"
            "    return { id, name, tags: [...tags] };\n"
            "  }",
            "register(name, tags = []) {\n"
            "    if (!name || !String(name).trim()) {\n"
            '      throw new Error("name required");\n'
            "    }\n"
            "    this._seq += 1;\n"
            "    const id = `svc-${this._seq}`;\n"
            "    // Broken: truncates name to 1 char\n"
            "    const short = String(name).slice(0, 1);\n"
            "    this.catalog.add(id, short, tags);\n"
            "    return { id, name: short, tags: [...tags] };\n"
            "  }",
        ),
        (
            "src/registry.js",
            "listByTag(tag) {\n    return this.catalog.findByTag(tag);\n  }",
            "listByTag(tag) {\n    // Broken: never delegates\n    void tag;\n    return [];\n  }",
        ),
    ),
    uses_not_implemented=False,
)

TS_FAULT_H = FaultPlan(
    description="add rejects all; seq decrements; size overcounts.",
    replacements=(
        (
            "src/catalog.js",
            "add(id, name, tags = []) {\n"
            "    if (!id || !name) {\n"
            '      throw new Error("id and name required");\n'
            "    }\n"
            "    this.items.set(id, { name, tags: [...tags] });\n"
            "  }",
            "add(id, name, tags = []) {\n"
            "    // Broken: always throws\n"
            "    void tags;\n"
            '    throw new Error("id and name required");\n'
            "  }",
        ),
        (
            "src/catalog.js",
            "size() {\n    return this.items.size;\n  }",
            "size() {\n    return this.items.size + 5;\n  }",
        ),
        (
            "src/registry.js",
            "register(name, tags = []) {\n"
            "    if (!name || !String(name).trim()) {\n"
            '      throw new Error("name required");\n'
            "    }\n"
            "    this._seq += 1;\n"
            "    const id = `svc-${this._seq}`;\n"
            "    this.catalog.add(id, name, tags);\n"
            "    return { id, name, tags: [...tags] };\n"
            "  }",
            "register(name, tags = []) {\n"
            "    if (!name || !String(name).trim()) {\n"
            '      throw new Error("name required");\n'
            "    }\n"
            "    // Broken: decrements seq\n"
            "    this._seq -= 1;\n"
            "    const id = `svc-${this._seq}`;\n"
            "    this.catalog.add(id, name, tags);\n"
            "    return { id, name, tags: [...tags] };\n"
            "  }",
        ),
    ),
    uses_not_implemented=False,
)

TS_FAULT_I = FaultPlan(
    description="findByTag returns names only; count -1; lookup forces id prefix.",
    replacements=(
        (
            "src/catalog.js",
            "findByTag(tag) {\n"
            "    const out = [];\n"
            "    for (const [id, meta] of this.items.entries()) {\n"
            "      if (meta.tags.includes(tag)) {\n"
            "        out.push({ id, ...meta });\n"
            "      }\n"
            "    }\n"
            "    return out;\n"
            "  }",
            "findByTag(tag) {\n"
            "    const out = [];\n"
            "    for (const [id, meta] of this.items.entries()) {\n"
            "      if (meta.tags.includes(tag)) {\n"
            "        out.push(meta.name);\n"
            "      }\n"
            "      void id;\n"
            "    }\n"
            "    return out;\n"
            "  }",
        ),
        (
            "src/registry.js",
            "count() {\n    return this.catalog.size();\n  }",
            "count() {\n    return this.catalog.size() - 1;\n  }",
        ),
        (
            "src/registry.js",
            "lookup(id) {\n"
            "    const hit = this.catalog.get(id);\n"
            "    if (!hit) {\n"
            "      return null;\n"
            "    }\n"
            "    return { id, ...hit };\n"
            "  }",
            "lookup(id) {\n"
            "    // Broken: rewrites requested id\n"
            "    const forced = `svc-0`;\n"
            "    const hit = this.catalog.get(forced);\n"
            "    if (!hit) {\n"
            "      return null;\n"
            "    }\n"
            "    return { id: forced, ...hit };\n"
            "  }",
        ),
    ),
    uses_not_implemented=False,
)

TS_FAULT_J = FaultPlan(
    description=("get throws on miss; register uses non-svc id scheme; size sums tags."),
    replacements=(
        (
            "src/catalog.js",
            "get(id) {\n    return this.items.get(id) ?? null;\n  }",
            "get(id) {\n"
            "    if (!this.items.has(id)) {\n"
            '      throw new Error("missing");\n'
            "    }\n"
            "    return this.items.get(id);\n"
            "  }",
        ),
        (
            "src/catalog.js",
            "size() {\n    return this.items.size;\n  }",
            "size() {\n"
            "    let n = 0;\n"
            "    for (const meta of this.items.values()) {\n"
            "      n += meta.tags.length;\n"
            "    }\n"
            "    return n;\n"
            "  }",
        ),
        (
            "src/registry.js",
            "register(name, tags = []) {\n"
            "    if (!name || !String(name).trim()) {\n"
            '      throw new Error("name required");\n'
            "    }\n"
            "    this._seq += 1;\n"
            "    const id = `svc-${this._seq}`;\n"
            "    this.catalog.add(id, name, tags);\n"
            "    return { id, name, tags: [...tags] };\n"
            "  }",
            "register(name, tags = []) {\n"
            "    if (!name || !String(name).trim()) {\n"
            '      throw new Error("name required");\n'
            "    }\n"
            "    this._seq += 1;\n"
            "    // Broken: non-service id scheme\n"
            "    const id = `id-${this._seq}`;\n"
            "    this.catalog.add(id, name, tags);\n"
            "    return { id, name, tags: [...tags] };\n"
            "  }",
        ),
    ),
    uses_not_implemented=False,
)

TS_FAULT_K = FaultPlan(
    description="add stores tags reversed empty join; listByTag null; count nullish.",
    replacements=(
        (
            "src/catalog.js",
            "add(id, name, tags = []) {\n"
            "    if (!id || !name) {\n"
            '      throw new Error("id and name required");\n'
            "    }\n"
            "    this.items.set(id, { name, tags: [...tags] });\n"
            "  }",
            "add(id, name, tags = []) {\n"
            "    if (!id || !name) {\n"
            '      throw new Error("id and name required");\n'
            "    }\n"
            "    // Broken: reverse tags and drop first\n"
            "    this.items.set(id, { name, tags: [...tags].reverse().slice(1) });\n"
            "  }",
        ),
        (
            "src/registry.js",
            "listByTag(tag) {\n    return this.catalog.findByTag(tag);\n  }",
            "listByTag(tag) {\n    void tag;\n    return null;\n  }",
        ),
        (
            "src/registry.js",
            "count() {\n    return this.catalog.size();\n  }",
            "count() {\n    return null;\n  }",
        ),
    ),
    uses_not_implemented=False,
)

TS_FAULT_L = FaultPlan(
    description="findByTag ignore; register empty tags forced; get overwrites name.",
    replacements=(
        (
            "src/catalog.js",
            "findByTag(tag) {\n"
            "    const out = [];\n"
            "    for (const [id, meta] of this.items.entries()) {\n"
            "      if (meta.tags.includes(tag)) {\n"
            "        out.push({ id, ...meta });\n"
            "      }\n"
            "    }\n"
            "    return out;\n"
            "  }",
            "findByTag(tag) {\n"
            "    // Broken: returns all items regardless of tag\n"
            "    const out = [];\n"
            "    for (const [id, meta] of this.items.entries()) {\n"
            "      void tag;\n"
            "      out.push({ id, ...meta });\n"
            "    }\n"
            "    return out;\n"
            "  }",
        ),
        (
            "src/catalog.js",
            "get(id) {\n    return this.items.get(id) ?? null;\n  }",
            "get(id) {\n"
            "    const hit = this.items.get(id);\n"
            "    if (!hit) return null;\n"
            "    // Broken: renames\n"
            "    return { name: 'x', tags: hit.tags };\n"
            "  }",
        ),
        (
            "src/registry.js",
            "register(name, tags = []) {\n"
            "    if (!name || !String(name).trim()) {\n"
            '      throw new Error("name required");\n'
            "    }\n"
            "    this._seq += 1;\n"
            "    const id = `svc-${this._seq}`;\n"
            "    this.catalog.add(id, name, tags);\n"
            "    return { id, name, tags: [...tags] };\n"
            "  }",
            "register(name, tags = []) {\n"
            "    if (!name || !String(name).trim()) {\n"
            '      throw new Error("name required");\n'
            "    }\n"
            "    this._seq += 1;\n"
            "    const id = `svc-${this._seq}`;\n"
            "    // Broken: force empty tags\n"
            "    this.catalog.add(id, name, []);\n"
            "    return { id, name, tags: [] };\n"
            "  }",
        ),
    ),
    uses_not_implemented=False,
)


def _base(seed_id: str) -> HarborMotorSeed:
    for s in MOTOR_SEEDS:
        if s.seed_id == seed_id:
            return s
    raise KeyError(seed_id)


def _variant(
    base_id: str,
    *,
    variant: str,
    fault: FaultPlan,
    base_commit: str,
    held_out: HeldOutTest | None = None,
    display_suffix: str = "",
) -> HarborMotorSeed:
    base = _base(base_id)
    title = base.display_name + (f" [{display_suffix}]" if display_suffix else f" [{variant}]")
    return replace(
        base,
        seed_id=f"{base.seed_id}_{variant}",
        display_name=title,
        base_commit=base_commit,
        fault=fault,
        held_out=held_out or base.held_out,
    )


def _synthetic_base_commit(prefix: str, index: int) -> str:
    """Unique 40-char hex-style motor base commit (hybrid ship rewrites to real SHA)."""
    body = f"{index:039x}"[-39:]
    return f"{prefix}{body}"


# Ordered multi-file fault menus cycled for M7–M10 scale bands.
_PY_FAULT_MENU: tuple[tuple[FaultPlan, str], ...] = (
    (PYTHON_ORDERS_FAULT, "discount-reserve"),
    (PYTHON_FAULT_B, "tax-partial-reserve"),
    (PYTHON_FAULT_C, "line-total-can-reserve"),
    (PYTHON_FAULT_D, "always-discount-empty-cart"),
    (PYTHON_FAULT_E, "tax-skip-always-fail"),
    (PYTHON_FAULT_F, "double-tax-first-line"),
    (PYTHON_FAULT_G, "strict-discount-no-release"),
    (PYTHON_FAULT_H, "unit-price-double-it"),
    (PYTHON_FAULT_I, "empty-snapshot-flip-accept"),
    (PYTHON_FAULT_J, "half-discount-half-stock"),
    (PYTHON_FAULT_K, "no-tax-always-can"),
    (PYTHON_FAULT_L, "qty-bump-release-mul"),
)
_GO_FAULT_MENU: tuple[tuple[FaultPlan, str], ...] = (
    (GO_KV_FAULT, "get-upsert"),
    (GO_FAULT_B, "set-delete-fetch"),
    (GO_FAULT_C, "size-put-remove"),
    (GO_FAULT_D, "ghost-get-null-upsert"),
    (GO_FAULT_E, "empty-set-double-size"),
    (GO_FAULT_F, "put-reversed-zero-upsert"),
    (GO_FAULT_G, "ok-by-value-const-size"),
    (GO_FAULT_H, "set-prefix-empty-put"),
    (GO_FAULT_I, "delete-wipe-double-put"),
    (GO_FAULT_J, "invert-ok-drop-value"),
    (GO_FAULT_K, "key-as-value-always-err"),
    (GO_FAULT_L, "wrong-delete-key-clear"),
)
_TS_FAULT_MENU: tuple[tuple[FaultPlan, str], ...] = (
    (TS_REGISTRY_FAULT, "tag-seq"),
    (TS_FAULT_B, "get-register-taint"),
    (TS_FAULT_C, "lookup-count"),
    (TS_FAULT_D, "name-as-id"),
    (TS_FAULT_E, "seq-start-10"),
    (TS_FAULT_F, "dup-find-zero-size"),
    (TS_FAULT_G, "name-only-empty-list"),
    (TS_FAULT_H, "add-throws-dec-seq"),
    (TS_FAULT_I, "names-only-forced-id"),
    (TS_FAULT_J, "throw-get-id-scheme"),
    (TS_FAULT_K, "tag-slice-null-count"),
    (TS_FAULT_L, "all-tags-force-empty"),
)

# Historical M9 band: 24 variants × 3 languages = 72 (>70 floor with headroom).
M9_SHIP_VARIANTS_PER_LANG = 24
# M10 DeepSWE-parity band: 40 variants × 3 languages = 120 (>113 floor with yield headroom).
M10_SHIP_VARIANTS_PER_LANG = 40


def _expand_ship_variants(
    *,
    base_id: str,
    prefix: str,
    menu: tuple[tuple[FaultPlan, str], ...],
    held_out: HeldOutTest,
    count: int,
) -> list[HarborMotorSeed]:
    """Cycle multi-file fault menus into unique ship seeds (scale without hand bloat)."""
    if count < 1:
        return []
    if not menu:
        raise ValueError(f"empty fault menu for {base_id}")
    out: list[HarborMotorSeed] = []
    for i in range(1, count + 1):
        fault, label = menu[(i - 1) % len(menu)]
        cycle = (i - 1) // len(menu)
        suffix = label if cycle == 0 else f"{label}-c{cycle + 1}"
        out.append(
            _variant(
                base_id,
                variant=f"v{i}",
                fault=fault,
                base_commit=_synthetic_base_commit(prefix, i),
                held_out=held_out,
                display_suffix=suffix,
            )
        )
    return out


SHIP_MOTOR_SEEDS: tuple[HarborMotorSeed, ...] = tuple(
    # Python M10 ×40
    _expand_ship_variants(
        base_id="harbor_python_orders",
        prefix="a",
        menu=_PY_FAULT_MENU,
        held_out=PYTHON_ORDERS_HELD_OUT,
        count=M10_SHIP_VARIANTS_PER_LANG,
    )
    # Go M10 ×40
    + _expand_ship_variants(
        base_id="harbor_go_kvstore",
        prefix="b",
        menu=_GO_FAULT_MENU,
        held_out=GO_KV_HELD_OUT,
        count=M10_SHIP_VARIANTS_PER_LANG,
    )
    # TypeScript M10 ×40
    + _expand_ship_variants(
        base_id="harbor_ts_registry",
        prefix="c",
        menu=_TS_FAULT_MENU,
        held_out=TS_REGISTRY_HELD_OUT,
        count=M10_SHIP_VARIANTS_PER_LANG,
    )
)


def list_ship_motor_seeds(*, language: str | None = None) -> list[HarborMotorSeed]:
    if language is None:
        return list(SHIP_MOTOR_SEEDS)
    code = language.strip().lower()
    if code in {"ts", "js", "javascript"}:
        code = "typescript"
    if code == "py":
        code = "python"
    return [s for s in SHIP_MOTOR_SEEDS if s.language == code]


def get_ship_motor_seed(seed_id: str) -> HarborMotorSeed:
    for s in SHIP_MOTOR_SEEDS:
        if s.seed_id == seed_id:
            return s
    raise KeyError(
        f"unknown ship motor seed_id={seed_id!r}; "
        f"known={sorted(x.seed_id for x in SHIP_MOTOR_SEEDS)}"
    )


__all__ = [
    "HarborMotorSeed",
    "M10_SHIP_VARIANTS_PER_LANG",
    "M9_SHIP_VARIANTS_PER_LANG",
    "SHIP_MOTOR_SEEDS",
    "get_ship_motor_seed",
    "list_ship_motor_seeds",
]
