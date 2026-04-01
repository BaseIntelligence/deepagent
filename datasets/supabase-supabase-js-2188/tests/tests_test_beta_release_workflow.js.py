const test = require('node:test')
const assert = require('node:assert/strict')
const pkg = require('../package.json')

test('package defines beta release script for CI workflow', () => {
  assert.equal(typeof pkg.scripts['release-beta'], 'string')
  assert.match(pkg.scripts['release-beta'], /scripts\/release-beta\.(ts|js)/)
})
