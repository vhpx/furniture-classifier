'use client';

import { ColumnDef } from '@tanstack/react-table';

import { DataTableColumnHeader } from '@/components/ui/custom/tables/data-table-column-header';
import moment from 'moment';
import { StorageObject } from '@/types/primitives/StorageObject';
import { Translate } from 'next-translate';
import { StorageObjectRowActions } from './row-actions';
import { formatBytes } from '@/utils/file-helper';
import { Check, X } from 'lucide-react';

export const storageObjectsColumns = (
  t: Translate,
  setStorageObject: (value: StorageObject | undefined) => void
): ColumnDef<StorageObject>[] => [
  {
    accessorKey: 'id',
    header: ({ column }) => (
      <DataTableColumnHeader column={column} title={t('id')} />
    ),
    cell: ({ row }) => (
      <div className="line-clamp-1 min-w-[8rem]">{row.getValue('id')}</div>
    ),
  },
  {
    accessorKey: 'name',
    header: ({ column }) => (
      <DataTableColumnHeader column={column} title={t('name')} />
    ),
    cell: ({ row }) => (
      <div className="min-w-[8rem] font-semibold">
        {row.getValue('name') ? (row.getValue('name') as string) : '-'}
      </div>
    ),
  },
  {
    accessorKey: 'size',
    header: ({ column }) => (
      <DataTableColumnHeader column={column} title={t('size')} />
    ),
    cell: ({ row }) => (
      <div className="min-w-[8rem]">
        {row.original?.metadata?.size !== undefined
          ? formatBytes(row.original.metadata.size)
          : '-'}
      </div>
    ),
  },
  {
    accessorKey: 'converted_size',
    header: ({ column }) => (
      <DataTableColumnHeader column={column} title={t('converted_size')} />
    ),
    cell: ({ row }) => (
      <div className="min-w-[8rem]">
        {row.original?.metadata?.size !== undefined
          ? formatBytes(row.original.metadata.size)
          : '-'}
      </div>
    ),
  },
  {
    accessorKey: 'converted',
    header: ({ column }) => (
      <DataTableColumnHeader column={column} title={t('converted')} />
    ),
    cell: ({ row }) => (
      <div>{row.getValue('converted') ? <Check /> : <X />}</div>
    ),
  },
  {
    accessorKey: 'classified',
    header: ({ column }) => (
      <DataTableColumnHeader column={column} title={t('classified')} />
    ),
    cell: ({ row }) => (
      <div>{row.getValue('classified') ? <Check /> : <X />}</div>
    ),
  },
  {
    accessorKey: 'created_at',
    header: ({ column }) => (
      <DataTableColumnHeader column={column} title={t('created_at')} />
    ),
    cell: ({ row }) => (
      <div className="min-w-[8rem]">
        {row.getValue('created_at')
          ? moment(row.getValue('created_at')).format('DD/MM/YYYY, HH:mm:ss')
          : '-'}
      </div>
    ),
  },
  {
    id: 'actions',
    cell: ({ row }) => (
      <StorageObjectRowActions row={row} setStorageObject={setStorageObject} />
    ),
  },
];
