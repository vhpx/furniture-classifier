create policy "Allow all access for authenticated users 13v9it7_0"
on "storage"."objects"
as permissive
for delete
to authenticated
using ((bucket_id = 'storage'::text));

create policy "Allow all access for authenticated users 13v9it7_1"
on "storage"."objects"
as permissive
for insert
to authenticated
with check ((bucket_id = 'storage'::text));

create policy "Allow all access for authenticated users 13v9it7_2"
on "storage"."objects"
as permissive
for update
to authenticated
using ((bucket_id = 'storage'::text));

create policy "Allow all access for authenticated users 13v9it7_3"
on "storage"."objects"
as permissive
for select
to authenticated
using ((bucket_id = 'storage'::text));

CREATE OR REPLACE FUNCTION get_storage_size()
RETURNS BIGINT AS $$
DECLARE
    total_size BIGINT;
BEGIN
        SELECT COALESCE(SUM((metadata->>'size')::BIGINT), 0) INTO total_size
        FROM storage.objects
        WHERE bucket_id = 'storage'
        AND owner IS NOT NULL
        AND metadata->>'size' IS NOT NULL;
        
        RETURN total_size;
END;
$$ LANGUAGE plpgsql;

-- create public.users table
create table "public"."users" (
    "id" uuid PRIMARY KEY NOT NULL,
    "email" text NOT NULL
);

alter table "public"."users" enable row level security;

create policy "Enable read access for all users"
on "public"."users"
as permissive
for select
to authenticated
using ((id = auth.uid()));

-- trigger insert/update/delete public.users.id based on auth.users.id
create or replace function sync_users_id()
returns trigger as $$
begin
    if TG_OP = 'INSERT' then
        insert into public.users (id, email) values (NEW.id, NEW.email);
    elsif TG_OP = 'UPDATE' then
        update public.users set id = NEW.id, email = NEW.email where id = OLD.id;
    elsif TG_OP = 'DELETE' then
        delete from public.users where id = OLD.id;
    end if;
    return null;
end;
$$ language plpgsql SECURITY DEFINER;

create trigger sync_users_id_insert
after insert or update or delete on auth.users
for each row
execute function sync_users_id();