import { expect, it, vi } from 'vitest';

it('loads auth utils without resolving the date-fns barrel entrypoint', async () => {
  const imported = [] as string[];
  const realImport = globalThis.__import__ ?? null;

  vi.resetModules();

  await vi.isolateModules(async () => {
    const dynamicImport = async (specifier: string) => {
      imported.push(specifier);
      return import(specifier);
    };

    // @ts-expect-error test hook
    globalThis.__import__ = dynamicImport;

    await import('../src/renderer/utils/auth/utils');
  });

  // @ts-expect-error cleanup
  globalThis.__import__ = realImport;

  expect(imported).not.toContain('date-fns');
});
