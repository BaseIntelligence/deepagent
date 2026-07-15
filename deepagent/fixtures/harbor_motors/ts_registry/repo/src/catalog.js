/**
 * Catalog stores item metadata for the registration service.
 */

export class Catalog {
  constructor() {
    /** @type {Map<string, {name: string, tags: string[]}>} */
    this.items = new Map();
  }

  /**
   * @param {string} id
   * @param {string} name
   * @param {string[]} [tags]
   */
  add(id, name, tags = []) {
    if (!id || !name) {
      throw new Error("id and name required");
    }
    this.items.set(id, { name, tags: [...tags] });
  }

  /** @param {string} id */
  get(id) {
    return this.items.get(id) ?? null;
  }

  /** @param {string} tag */
  findByTag(tag) {
    const out = [];
    for (const [id, meta] of this.items.entries()) {
      if (meta.tags.includes(tag)) {
        out.push({ id, ...meta });
      }
    }
    return out;
  }

  size() {
    return this.items.size;
  }
}
