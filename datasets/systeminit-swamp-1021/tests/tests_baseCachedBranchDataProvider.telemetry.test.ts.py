import { BaseCachedBranchDataProvider } from '../src/tree/BaseCachedBranchDataProvider';

describe('BaseCachedBranchDataProvider telemetry provider names', () => {
    class AnonymousAzureProvider extends BaseCachedBranchDataProvider<any> {
        protected get contextValue(): string {
            return 'cosmosDB.azure';
        }

        protected createResourceItem(): undefined {
            return undefined;
        }

        protected onResourceItemRetrieved(): void {
            // no-op
        }
    }

    class AnonymousWorkspaceProvider extends BaseCachedBranchDataProvider<any> {
        protected get contextValue(): string {
            return 'cosmosDB.workspace';
        }

        protected createResourceItem(): undefined {
            return undefined;
        }

        protected onResourceItemRetrieved(): void {
            // no-op
        }
    }

    it('uses stable telemetry name for azure resources providers', () => {
        const provider = new AnonymousAzureProvider();

        expect((provider as any).providerName).toBe('CosmosDBBranchDataProvider');
    });

    it('uses stable telemetry name for workspace providers', () => {
        const provider = new AnonymousWorkspaceProvider();

        expect((provider as any).providerName).toBe('CosmosDBWorkspaceBranchDataProvider');
    });
});
