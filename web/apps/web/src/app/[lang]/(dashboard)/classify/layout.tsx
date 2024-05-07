import { ReactNode } from 'react';

export const dynamic = 'force-dynamic';

interface LayoutProps {
  children: ReactNode;
}

export default async function Layout({ children }: LayoutProps) {
  return <div className="p-4 pt-2 md:px-8 lg:px-16 xl:px-32">{children}</div>;
}
