import React from "react";

export default function Footer() {
  return (
    <footer className="shadow-sm">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-4">
        <p className="text-sm" style={{ color: '#E63946'}}>
         Marco Cabrera Hernández © {new Date().getFullYear()} All rights reserved.
        </p>
      </div>
    </footer>
  );
}
