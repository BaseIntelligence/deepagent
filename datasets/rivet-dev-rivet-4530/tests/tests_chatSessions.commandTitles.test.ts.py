/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/

import assert from 'assert';
import { ensureNoDisposablesAreLeakedInTestSuite } from '../src/vs/base/test/common/utils.js';
import { MenuRegistry } from '../src/vs/platform/actions/common/actions.js';
import { registerChatActions } from '../src/vs/workbench/contrib/chat/browser/actions/chatActions.js';
import { registerChatExecuteActions } from '../src/vs/workbench/contrib/chat/browser/actions/chatExecuteActions.js';
import { registerChatTitleActions } from '../src/vs/workbench/contrib/chat/browser/actions/chatTitleActions.js';
import { registerQuickChatActions } from '../src/vs/workbench/contrib/chat/browser/actions/quickChatActions.js';
import '../src/vs/workbench/contrib/chat/browser/chatSessions/chatSessions.contribution.js';

suite('Chat session provider command titles', () => {
	let registered = false;

	function ensureActionsRegistered(): void {
		if (!registered) {
			registerChatActions();
			registerChatExecuteActions();
			registerChatTitleActions();
			registerQuickChatActions();
			registered = true;
		}
	}

	setup(() => {
		ensureActionsRegistered();
	});

	ensureNoDisposablesAreLeakedInTestSuite();

	for (const [provider, providerName] of [
		['local', 'Local'],
		['copilotcli', 'Copilot CLI'],
	] as const) {
		test(`uses session wording for ${provider}`, () => {
			const command = MenuRegistry.getCommand(`workbench.action.chat.openNewChatSessionInPlace.${provider}`);
			assert.ok(command, `expected command for ${provider} to be registered`);
			assert.strictEqual(command.title.value, `New ${providerName} Session`);
		});
	}
});
