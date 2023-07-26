"use client";

import { usePathname } from "next/navigation";

const navigation = [{ name: "Home", href: "/" }];

function classNames(...classes: string[]) {
  return classes.filter(Boolean).join(" ");
}

export default function Header() {
  const pathname = usePathname();

  return (
    <>
      <div className="mx-auto max-w-7xl px-5 sm:px-6 lg:px-8">
        <div className="flex h-16 justify-between">
          <div className="flex">
            <div className="hidden sm:-my-px sm:ml-6 sm:flex sm:space-x-8">
              {navigation.map((item) => (
                <a
                  key={item.name}
                  href={item.href}
                  style={{ color: '#E63946', fontWeight: 'bold' }}
                  className={classNames(
                    pathname === item.href
                      ? "border-slate-500"
                      : "border-transparent",
                    "inline-flex items-center font-medium"
                  )
                }
                  aria-current={pathname === item.href ? "page" : undefined}
                >
                  {item.name}
                </a>
              ))}
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
