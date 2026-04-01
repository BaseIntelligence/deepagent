import React from 'react'
import { render, screen } from '@testing-library/react'

jest.mock('@stump/client', () => ({
	useGraphQL: jest.fn(),
}))

jest.mock('@stump/i18n', () => ({
	useLocaleContext: jest.fn(),
}))

jest.mock('@/paths', () => ({
	usePaths: jest.fn(),
}))

jest.mock('../../packages/browser/src/scenes/library/tabs/settings/context', () => ({
	useLibraryManagement: jest.fn(),
}))

jest.mock('../../packages/browser/src/components/table', () => ({
	Table: ({ data, columns }: any) => (
		<div data-testid="missing-entities-table">
			<div data-testid="row-count">{data.length}</div>
			<div>{String(columns[0].header)}</div>
		</div>
	),
}))

jest.mock('@stump/components', () => ({
	Badge: ({ children }: any) => <span>{children}</span>,
	Card: ({ children }: any) => <div>{children}</div>,
	Heading: ({ children }: any) => <div>{children}</div>,
	Link: ({ children, to }: any) => <a href={to}>{children}</a>,
	Text: ({ children }: any) => <div>{children}</div>,
}))

jest.mock('@tanstack/react-query', () => ({
	keepPreviousData: undefined,
}))

const mockUseGraphQL = jest.requireMock('@stump/client').useGraphQL as jest.Mock
const mockUseLocaleContext = jest.requireMock('@stump/i18n').useLocaleContext as jest.Mock
const mockUsePaths = jest.requireMock('@/paths').usePaths as jest.Mock
const mockUseLibraryManagement = jest.requireMock('../../packages/browser/src/scenes/library/tabs/settings/context')
	.useLibraryManagement as jest.Mock

describe('MissingEntitiesTable', () => {
	beforeEach(() => {
		jest.resetModules()
		mockUseLocaleContext.mockReturnValue({ t: (key: string) => key })
		mockUsePaths.mockReturnValue({
			bookOverview: (id: string) => `/books/${id}`,
			seriesOverview: (id: string) => `/series/${id}`,
		})
		mockUseLibraryManagement.mockReturnValue({
			library: { id: 'library-1' },
		})
		mockUseGraphQL.mockReturnValue({
			data: {
				libraryMissingEntities: {
					nodes: [
						{ id: 'book-1', path: '/missing/book.cbz', type: 'BOOK' },
						{ id: 'series-1', path: '/missing/series', type: 'SERIES' },
					],
					pageInfo: {
						__typename: 'OffsetPaginationInfo',
						totalPages: 1,
						currentPage: 1,
						pageSize: 10,
						pageOffset: 0,
						zeroBased: false,
						totalItems: 2,
					},
				},
			},
			isLoading: false,
		})
	})

	it('renders a table for missing library entities when results exist', () => {
		const { default: MissingEntitiesTable } = require('../../packages/browser/src/scenes/library/tabs/settings/danger/deletion/MissingEntitiesTable')

		render(<MissingEntitiesTable />)

		expect(screen.getByTestId('missing-entities-table')).toBeInTheDocument()
		expect(screen.getByTestId('row-count')).toHaveTextContent('2')
		expect(mockUseGraphQL).toHaveBeenCalled()
	})
})
