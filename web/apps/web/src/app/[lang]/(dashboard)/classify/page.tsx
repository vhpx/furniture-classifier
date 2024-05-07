import { createServerComponentClient } from '@supabase/auth-helpers-nextjs';
import { cookies } from 'next/headers';
import { StorageObject } from '@/types/primitives/StorageObject';
import StorageObjectsTable from './table';
import useTranslation from 'next-translate/useTranslation';
import { Separator } from '@/components/ui/separator';
import { formatBytes } from '@/utils/file-helper';
import { getCurrentUser } from '@/lib/user-helper';
import { redirect } from 'next/navigation';

interface Props {
  searchParams: {
    q: string;
    page: string;
    pageSize: string;
  };
}

export default async function WorkspaceStorageObjectsPage({
  searchParams,
}: Props) {
  const { t } = useTranslation('ws-storage-objects');

  const user = await getCurrentUser();
  const { data, count } = await getData(searchParams);

  const totalSize = await getTotalSize();
  const largestFile = await getLargestFile();
  const smallestFile = await getSmallestFile();

  if (!user) redirect('/login');

  return (
    <>
      <div className="mb-8 mt-4 grid gap-4 text-center md:grid-cols-2 xl:grid-cols-4">
        <div className="border-border bg-foreground/5 rounded-lg border p-4">
          <h2 className="text-lg font-semibold">{t('total_files')}</h2>
          <Separator className="my-2" />
          <div className="text-3xl font-bold">{count}</div>
        </div>

        <div className="border-border bg-foreground/5 rounded-lg border p-4">
          <h2 className="text-lg font-semibold">{t('total_size')}</h2>
          <Separator className="my-2" />
          <div className="text-3xl font-bold">{formatBytes(totalSize)}</div>
        </div>

        <div className="border-border bg-foreground/5 rounded-lg border p-4">
          <h2 className="text-lg font-semibold">{t('largest_file')}</h2>
          <Separator className="my-2" />
          <div className="text-3xl font-bold">
            {data.length > 0 ? formatBytes(largestFile?.size as number) : '-'}
          </div>
        </div>

        <div className="border-border bg-foreground/5 rounded-lg border p-4">
          <h2 className="text-lg font-semibold">{t('smallest_file')}</h2>
          <Separator className="my-2" />
          <div className="text-3xl font-bold">
            {data.length > 0 ? formatBytes(smallestFile?.size as number) : '-'}
          </div>
        </div>
      </div>

      <StorageObjectsTable data={data} count={count} />
    </>
  );
}

async function getData({
  q,
  page = '1',
  pageSize = '10',
}: {
  q?: string;
  page?: string;
  pageSize?: string;
}) {
  const supabase = createServerComponentClient({ cookies });

  const queryBuilder = supabase
    .schema('storage')
    .from('objects')
    .select('*', {
      count: 'exact',
    })
    .eq('bucket_id', 'storage')
    .not('owner', 'is', null)
    .order('created_at', { ascending: false });

  if (q) queryBuilder.ilike('name', `%${q}%`);

  if (page && pageSize) {
    const parsedPage = parseInt(page);
    const parsedSize = parseInt(pageSize);
    const start = (parsedPage - 1) * parsedSize;
    const end = parsedPage * parsedSize;
    queryBuilder.range(start, end).limit(parsedSize);
  }

  const { data, error, count } = await queryBuilder;
  if (error) throw error;

  return { data, count } as {
    data: StorageObject[];
    count: number;
  };
}

async function getTotalSize() {
  const supabase = createServerComponentClient({ cookies });

  const { data, error } = await supabase.rpc('get_storage_size');

  if (error) throw error;
  return data;
}

async function getLargestFile() {
  const supabase = createServerComponentClient({ cookies });

  const { data, error } = await supabase
    .schema('storage')
    .from('objects')
    .select('metadata->size')
    .eq('bucket_id', 'storage')
    .not('owner', 'is', null)
    .order('metadata->size', { ascending: false })
    .limit(1)
    .maybeSingle();

  if (error) throw error;
  return data;
}

async function getSmallestFile() {
  const supabase = createServerComponentClient({ cookies });

  const { data, error } = await supabase
    .schema('storage')
    .from('objects')
    .select('metadata->size')
    .eq('bucket_id', 'storage')
    .not('owner', 'is', null)
    .order('metadata->size', { ascending: true })
    .limit(1)
    .maybeSingle();

  if (error) throw error;
  return data;
}
