import { Separator } from '@/components/ui/separator';
import useTranslation from 'next-translate/useTranslation';

export default async function Footer() {
  const { t } = useTranslation('common');

  return (
    <div className="w-full text-center">
      <Separator className="bg-foreground/5 mt-8" />
      <div className="p-4 text-center opacity-80 md:px-32 xl:px-64">
        {t('copyright')}
      </div>
    </div>
  );
}
