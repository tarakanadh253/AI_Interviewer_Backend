import { useLocation, Link } from "react-router-dom";
import { useEffect } from "react";
import { Button } from "@/components/ui/button";

const NotFound = () => {
  const location = useLocation();

  useEffect(() => {
    console.error("404 Error: User attempted to access non-existent route:", location.pathname);
  }, [location.pathname]);

  return (
    <div className="flex min-h-screen items-center justify-center bg-background text-foreground relative overflow-hidden">
      {/* Ambience */}
      <div className="absolute top-0 right-0 w-[500px] h-[500px] bg-ohg-navy/5 rounded-full blur-[100px] pointer-events-none -translate-y-1/2 translate-x-1/2" />
      <div className="absolute bottom-0 left-0 w-[500px] h-[500px] bg-ohg-orange/5 rounded-full blur-[100px] pointer-events-none translate-y-1/2 -translate-x-1/2" />

      <div className="text-center relative z-10">
        <h1 className="mb-4 text-8xl font-bold text-ohg-navy">404</h1>
        <p className="mb-8 text-2xl text-muted-foreground font-medium">Page not found</p>
        <p className="mb-8 text-muted-foreground">The page you are looking for doesn't exist or has been moved.</p>
        <Link to="/">
          <Button className="bg-ohg-navy hover:bg-ohg-navy/90 text-white px-8 py-6 rounded-full text-lg font-bold">
            Return to Home
          </Button>
        </Link>
      </div>
    </div>
  );
};

export default NotFound;
