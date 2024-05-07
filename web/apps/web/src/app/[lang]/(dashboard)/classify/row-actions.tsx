'use client';

import { DotsHorizontalIcon } from '@radix-ui/react-icons';
import { Row } from '@tanstack/react-table';

import { Button } from '@/components/ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { useRouter } from 'next/navigation';
import { toast } from '@/components/ui/use-toast';
import { StorageObject } from '@/types/primitives/StorageObject';
import useTranslation from 'next-translate/useTranslation';
import { createClientComponentClient } from '@supabase/auth-helpers-nextjs';

interface Props {
  row: Row<StorageObject>;
  setStorageObject: (value: StorageObject | undefined) => void;
}

export function StorageObjectRowActions(props: Props) {
  const supabase = createClientComponentClient();
  const { t } = useTranslation();

  const router = useRouter();
  const storageObj = props.row.original;

  const deleteStorageObject = async () => {
    if (!storageObj.name) return;

    const { error } = await supabase.storage
      .from('storage')
      .remove([storageObj.name]);

    if (!error) {
      router.refresh();
    } else {
      toast({
        title: 'Failed to delete file',
        description: error.message,
      });
    }
  };

  const renameStorageObject = async () => {
    if (!storageObj.name) return;

    const newName = prompt('Enter new name', storageObj.name);

    if (!newName) return;

    // re-add extension if it was removed
    const safeNewName = storageObj.name.includes('.')
      ? newName.includes('.')
        ? newName
        : `${newName}.${storageObj.name.split('.').pop()}`
      : newName;

    const { error } = await supabase.storage
      .from('storage')
      .move(storageObj.name, safeNewName);

    if (!error) {
      router.refresh();
    } else {
      toast({
        title: 'Failed to rename file',
        description: error.message,
      });
    }
  };

  const downloadStorageObject = async () => {
    if (!storageObj.name) return;

    const { data, error } = await supabase.storage
      .from('storage')
      .download(storageObj.name);

    if (error) {
      toast({
        title: 'Failed to download file',
        description: error.message,
      });
      return;
    }

    const url = URL.createObjectURL(data);

    const a = document.createElement('a');
    a.href = url;
    a.download = storageObj.name || '';
    a.click();

    URL.revokeObjectURL(url);
  };

  if (!storageObj.id) return null;

  return (
    <>
      <DropdownMenu modal={false}>
        <DropdownMenuTrigger asChild>
          <Button
            variant="ghost"
            className="data-[state=open]:bg-muted flex h-8 w-8 p-0"
          >
            <DotsHorizontalIcon className="h-4 w-4" />
            <span className="sr-only">Open menu</span>
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="w-[160px]">
          <DropdownMenuItem onClick={renameStorageObject}>
            {t('common:rename')}
          </DropdownMenuItem>
          <DropdownMenuItem onClick={downloadStorageObject}>
            {t('common:download')}
          </DropdownMenuItem>
          <DropdownMenuSeparator />
          <DropdownMenuItem onClick={deleteStorageObject}>
            {t('common:delete')}
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </>
  );
}
