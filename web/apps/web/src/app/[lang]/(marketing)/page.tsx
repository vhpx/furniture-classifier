import useTranslation from 'next-translate/useTranslation';
import GradientHeadline from './gradient-headline';
import GetStartedButton from './get-started-button';
import { getCurrentUser } from '@/lib/user-helper';
import { Suspense } from 'react';

export const dynamic = 'force-dynamic';

export default async function MarketingPage() {
  const { t } = useTranslation('home');

  const user = await getCurrentUser();

  return (
    <div className="flex min-h-full w-full flex-col items-center">
      <div className="text-foreground mt-8 flex max-w-6xl flex-col gap-6 px-3 py-16 lg:gap-14 lg:py-24">
        <div className="mb-4 flex flex-col items-center lg:mb-12">
          <h1 className="relative mb-4 text-center font-mono text-4xl font-bold lg:text-7xl">
            Furniture
            <br />
            Classifier
          </h1>

          <p className="mx-auto my-4 max-w-xl text-center text-lg font-semibold !leading-tight md:mb-8 md:text-2xl lg:text-3xl">
            {t('headline-p1')} <br />
            <GradientHeadline title={t('headline-p2')} />.
          </p>

          <Suspense fallback={<GetStartedButton href="/login" />}>
            <GetStartedButton href={user ? '/classify' : '/login'} />
          </Suspense>
        </div>
      </div>
    </div>
  );
}
