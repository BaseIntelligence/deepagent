const fs = require('fs')
const path = require('path')
const test = require('node:test')
const assert = require('node:assert/strict')

const workflowPath = path.join(process.cwd(), '.github', 'workflows', 'publish.yml')

function readWorkflow() {
  return fs.readFileSync(workflowPath, 'utf8')
}

test('publish workflow exposes beta release job and beta input', () => {
  const workflow = readWorkflow()

  assert.match(workflow, /^\s*release-beta:/m)
  assert.match(workflow, /beta/i)
  assert.match(workflow, /version_specifier:/)
})

test('package scripts include beta release command used by CI', () => {
  const packageJson = JSON.parse(fs.readFileSync(path.join(process.cwd(), 'package.json'), 'utf8'))

  assert.ok(packageJson.scripts['release-beta'], 'expected package.json to define release-beta script')
})
