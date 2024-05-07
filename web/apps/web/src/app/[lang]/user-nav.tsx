import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuLabel,
  DropdownMenuPortal,
  DropdownMenuSeparator,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { getCurrentUser } from '@/lib/user-helper';
import { Globe, Palette, User } from 'lucide-react';
import { Suspense } from 'react';
import { LogoutDropdownItem } from './(dashboard)/_components/logout-dropdown-item';
import { ThemeDropdownItems } from './(dashboard)/_components/theme-dropdown-items';
import { LanguageWrapper } from './(dashboard)/_components/language-wrapper';
import useTranslation from 'next-translate/useTranslation';
import { SystemLanguageWrapper } from './(dashboard)/_components/system-language-wrapper';
import DashboardMenuItem from './dashboard-menu-item';

export async function UserNav() {
  const { t } = useTranslation('common');

  const user = await getCurrentUser();

  return (
    <DropdownMenu modal={false}>
      <DropdownMenuTrigger asChild>
        <Avatar className="relative cursor-pointer overflow-visible font-semibold">
          <Suspense fallback={<AvatarFallback>?</AvatarFallback>}>
            <AvatarImage className="overflow-clip rounded-full" />
            <AvatarFallback className="font-semibold">
              <User className="h-5 w-5" />
            </AvatarFallback>
            <div className="border-background absolute bottom-0 right-0 z-20 h-3 w-3 rounded-full border-2 bg-green-500 dark:bg-green-400" />
          </Suspense>
        </Avatar>
      </DropdownMenuTrigger>
      <DropdownMenuContent className="w-56" align="end" forceMount>
        <DropdownMenuLabel className="font-normal">
          <div className="flex flex-col">
            <div className="line-clamp-1 w-fit break-all text-sm font-medium hover:underline">
              {user?.email}
            </div>
          </div>
        </DropdownMenuLabel>
        <DashboardMenuItem />
        <DropdownMenuSeparator />
        <DropdownMenuGroup>
          <DropdownMenuSub>
            <DropdownMenuSubTrigger>
              <Palette className="mr-2 h-4 w-4" />
              <span>{t('theme')}</span>
            </DropdownMenuSubTrigger>
            <DropdownMenuPortal>
              <DropdownMenuSubContent>
                <ThemeDropdownItems />
              </DropdownMenuSubContent>
            </DropdownMenuPortal>
          </DropdownMenuSub>
          <DropdownMenuSub>
            <DropdownMenuSubTrigger>
              <Globe className="mr-2 h-4 w-4" />
              <span>{t('language')}</span>
            </DropdownMenuSubTrigger>
            <DropdownMenuPortal>
              <DropdownMenuSubContent>
                <LanguageWrapper locale="en" label="English" />
                <LanguageWrapper locale="vi" label="Tiếng Việt" />
                <DropdownMenuSeparator />
                <SystemLanguageWrapper />
              </DropdownMenuSubContent>
            </DropdownMenuPortal>
          </DropdownMenuSub>
        </DropdownMenuGroup>
        <DropdownMenuSeparator />
        <LogoutDropdownItem />
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
