import { existsSync, mkdirSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { checkStaleness } from '../packages/intent/src/staleness.js'

let tmpDir: string
const originalFetch = globalThis.fetch

function setupDir(): string {
  const dir = join(
    tmpdir(),
    `intent-local-version-test-${Date.now()}-${Math.random().toString(36).slice(2)}`,
  )
  mkdirSync(dir, { recursive: true })
  return dir
}

function writeSkill(
  dir: string,
  name: string,
  fm: Record<string, unknown>,
  body = '# Skill\n',
): void {
  const skillDir = join(dir, 'skills', ...name.split('/'))
  mkdirSync(skillDir, { recursive: true })

  const fmStr = Object.entries(fm)
    .map(([k, v]) => `${k}: ${v}`)
    .join('\n')

  writeFileSync(join(skillDir, 'SKILL.md'), `---\n${fmStr}\n---\n${body}`)
}

function mockFetchVersion(version: string): void {
  globalThis.fetch = vi.fn().mockResolvedValue({
    ok: true,
    json: () => Promise.resolve({ version }),
  } as Response)
}

function mockFetchNotOk(): void {
  globalThis.fetch = vi.fn().mockResolvedValue({ ok: false } as Response)
}

beforeEach(() => {
  tmpDir = setupDir()
})

afterEach(() => {
  globalThis.fetch = originalFetch
  if (existsSync(tmpDir)) {
    rmSync(tmpDir, { recursive: true, force: true })
  }
})

describe('local package.json version resolution', () => {
  it('uses local package.json version when registry lookup fails', async () => {
    writeFileSync(
      join(tmpDir, 'package.json'),
      JSON.stringify({ name: '@private/lib', version: '2.5.0' }),
    )

    writeSkill(tmpDir, 'core', {
      name: 'core',
      description: 'Core',
      library_version: '2.0.0',
    })

    mockFetchNotOk()

    const report = await checkStaleness(tmpDir, '@private/lib')

    expect(report.currentVersion).toBe('2.5.0')
    expect(report.versionDrift).toBe('minor')
    expect(report.skills[0]?.needsReview).toBe(true)
  })

  it('prefers local package.json version over registry version', async () => {
    writeFileSync(
      join(tmpDir, 'package.json'),
      JSON.stringify({ name: '@example/lib', version: '3.0.0' }),
    )

    writeSkill(tmpDir, 'core', {
      name: 'core',
      description: 'Core',
      library_version: '2.0.0',
    })

    mockFetchVersion('2.5.0')

    const report = await checkStaleness(tmpDir, '@example/lib')

    expect(report.currentVersion).toBe('3.0.0')
    expect(report.versionDrift).toBe('major')
  })
})
