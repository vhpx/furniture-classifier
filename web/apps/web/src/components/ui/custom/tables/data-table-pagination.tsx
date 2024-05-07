import {
  ChevronLeftIcon,
  ChevronRightIcon,
  DoubleArrowLeftIcon,
  DoubleArrowRightIcon,
} from '@radix-ui/react-icons';
import { Table } from '@tanstack/react-table';

import { Button } from '@/components/ui/button';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import useQuery from '@/lib/hooks/use-query';
import { cn } from '@/lib/utils';
import useTranslation from 'next-translate/useTranslation';
import { Separator } from '../../separator';
import { toLower } from 'lodash';

interface DataTablePaginationProps<TData> {
  table: Table<TData>;
  count?: number;
  className?: string;
}

export function DataTablePagination<TData>({
  table,
  count,
  className,
}: DataTablePaginationProps<TData>) {
  const { t } = useTranslation('common');
  const query = useQuery();

  const sizes = [5, 10, 20, 50, 100, 200, 500, 1000];
  const isPageOutOfRange =
    table.getState().pagination.pageIndex + 1 > table.getPageCount();

  return (
    <div
      className={cn(
        'flex flex-col items-center justify-between gap-1 px-2 text-center md:flex-row',
        className
      )}
    >
      {count !== undefined && count > 0 ? (
        <div className="text-muted-foreground flex-none text-sm">
          <span className="text-primary font-semibold">{count}</span>{' '}
          {toLower(count === 1 ? t('row') : t('rows'))}.
        </div>
      ) : (
        <div />
      )}

      <Separator className="my-1 md:hidden" />

      <div className="flex flex-wrap items-center justify-center gap-2 text-center md:gap-4 lg:gap-8">
        <div className="hidden items-center space-x-2 md:flex">
          <p className="text-sm font-medium">{t('rows-per-page')}</p>
          <Select
            value={`${table.getState().pagination.pageSize}`}
            onValueChange={(value) => {
              table.setPageIndex(0);
              table.setPageSize(Number(value));
              query.set({ page: 1, pageSize: value });
            }}
          >
            <SelectTrigger className="h-8 w-[70px]">
              <SelectValue placeholder={table.getState().pagination.pageSize} />
            </SelectTrigger>
            <SelectContent side="top">
              {sizes.map((pageSize) => (
                <SelectItem key={pageSize} value={`${pageSize}`}>
                  {pageSize}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className="text-muted-foreground w-fit text-sm">
          {t('page')}{' '}
          <span className="text-primary font-semibold">
            {isPageOutOfRange ? 1 : table.getState().pagination.pageIndex + 1}
          </span>{' '}
          {t('of')}{' '}
          <span className="text-primary font-semibold">
            {table.getPageCount()}
          </span>
        </div>
        <div className="flex items-center space-x-2">
          <Button
            variant="outline"
            className="hidden h-8 w-8 p-0 lg:flex"
            onClick={() => {
              table.resetRowSelection();
              table.setPageIndex(0);
              query.set({ page: 1 });
            }}
            disabled={!table.getCanPreviousPage() || isPageOutOfRange}
          >
            <span className="sr-only">Go to first page</span>
            <DoubleArrowLeftIcon className="h-4 w-4" />
          </Button>
          <Button
            variant="outline"
            className="h-8 w-8 p-0"
            onClick={() => {
              table.resetRowSelection();
              table.previousPage();
              query.set({ page: table.getState().pagination.pageIndex });
            }}
            disabled={!table.getCanPreviousPage() || isPageOutOfRange}
          >
            <span className="sr-only">Go to previous page</span>
            <ChevronLeftIcon className="h-4 w-4" />
          </Button>
          <Button
            variant="outline"
            className="h-8 w-8 p-0"
            onClick={() => {
              table.resetRowSelection();
              table.nextPage();
              query.set({
                page: isPageOutOfRange
                  ? 2
                  : table.getState().pagination.pageIndex + 2,
              });
            }}
            disabled={!table.getCanNextPage() && !isPageOutOfRange}
          >
            <span className="sr-only">Go to next page</span>
            <ChevronRightIcon className="h-4 w-4" />
          </Button>
          <Button
            variant="outline"
            className="hidden h-8 w-8 p-0 lg:flex"
            onClick={() => {
              table.resetRowSelection();
              table.setPageIndex(table.getPageCount() - 1);
              query.set({ page: table.getPageCount() });
            }}
            disabled={!table.getCanNextPage() && !isPageOutOfRange}
          >
            <span className="sr-only">Go to last page</span>
            <DoubleArrowRightIcon className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </div>
  );
}
