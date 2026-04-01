// Swamp, an Automation Framework
// Copyright (C) 2026 System Initiative, Inc.
//
// This file is part of Swamp.
//
// Swamp is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License version 3
// as published by the Free Software Foundation, with the Swamp
// Extension and Definition Exception (found in the "COPYING-EXCEPTION"
// file).
//
// Swamp is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with Swamp.  If not, see <https://www.gnu.org/licenses/>.

import { assertEquals } from "@std/assert";
import { DefaultDatastorePathResolver } from "../src/infrastructure/persistence/default_datastore_path_resolver.ts";
import { createModelOutputGetDeps } from "../src/libswamp/models/output_get.ts";
import { createModelEvaluateDeps } from "../src/libswamp/models/evaluate.ts";
import { ModelType } from "../src/domain/models/model_type.ts";

Deno.test("model output deps use datastore cache path for outputs", () => {
  const repoDir = "/repo";
  const resolver = new DefaultDatastorePathResolver(repoDir, {
    type: "s3",
    config: { bucket: "bucket" },
    datastorePath: "/remote/datastore",
    cachePath: "/tmp/swamp-cache",
  });

  const deps = createModelOutputGetDeps(repoDir, resolver);
  const outputRepo = (deps as unknown as {
    outputRepo?: { getPath: (type: ModelType, method: string, id: string) => string };
  }).outputRepo;

  const path = outputRepo?.getPath(ModelType.create("server"), "deploy", "abc");

  assertEquals(path?.startsWith("/tmp/swamp-cache/outputs/"), true);
});

Deno.test("model evaluate deps use datastore cache path for evaluated definitions", () => {
  const repoDir = "/repo";
  const resolver = new DefaultDatastorePathResolver(repoDir, {
    type: "s3",
    config: { bucket: "bucket" },
    datastorePath: "/remote/datastore",
    cachePath: "/tmp/swamp-cache",
  });

  const deps = createModelEvaluateDeps(repoDir, resolver);
  const path = deps.getEvaluatedPath(ModelType.create("server"), "def-123" as never);

  assertEquals(path.startsWith("/tmp/swamp-cache/definitions-evaluated/"), true);
});
