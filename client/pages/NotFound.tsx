import { useLocation } from "react-router-dom";
import { useEffect } from "react";

const NotFound = () => {
  const location = useLocation();

  useEffect(() => {
    console.error(
      "404 Error: User attempted to access non-existent route:",
      location.pathname,
    );
  }, [location.pathname]);

  return (
    <div className="min-h-screen flex items-center justify-center">
      <div className="text-center">
        <h1 className="text-7xl font-extrabold text-white mb-2">404</h1>
        <p className="text-white/70 mb-6">Oops! Page not found</p>
        <a href="/" className="inline-block rounded-full border border-white/10 bg-white/5 px-4 py-2 text-sm text-white hover:bg-white/10">
          Return to Home
        </a>
      </div>
    </div>
  );
};

export default NotFound;
