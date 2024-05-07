import Footer from './footer';
import React from 'react';

interface LayoutProps {
  children: React.ReactNode;
}

export default async function Layout({ children }: LayoutProps) {
  return (
    <>
      {children}
      <Footer />
    </>
  );
}
