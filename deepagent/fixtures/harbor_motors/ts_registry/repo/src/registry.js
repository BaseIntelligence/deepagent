/**
 * Registry orchestrates catalog + counter for named service registration.
 */

import { Catalog } from "./catalog.js";

export class Registry {
  /**
   * @param {Catalog} [catalog]
   */
  constructor(catalog = new Catalog()) {
    this.catalog = catalog;
    this._seq = 0;
  }

  /**
   * @param {string} name
   * @param {string[]} [tags]
   * @returns {{id: string, name: string, tags: string[]}}
   */
  register(name, tags = []) {
    if (!name || !String(name).trim()) {
      throw new Error("name required");
    }
    this._seq += 1;
    const id = `svc-${this._seq}`;
    this.catalog.add(id, name, tags);
    return { id, name, tags: [...tags] };
  }

  /** @param {string} id */
  lookup(id) {
    const hit = this.catalog.get(id);
    if (!hit) {
      return null;
    }
    return { id, ...hit };
  }

  /** @param {string} tag */
  listByTag(tag) {
    return this.catalog.findByTag(tag);
  }

  count() {
    return this.catalog.size();
  }
}
