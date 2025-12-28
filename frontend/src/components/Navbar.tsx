import { useState, useEffect } from "react";
import { Link, useNavigate } from "react-router-dom";
import {
    Search,
    Menu as MenuIcon,
    ChevronDown,
    Terminal,
    X
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuItem,
    DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Sheet, SheetContent, SheetTrigger, SheetClose } from "@/components/ui/sheet";

export const Navbar = () => {
    const navigate = useNavigate();


    return (
        <nav className="w-full bg-background border-b border-border shadow-sm sticky top-0 z-50 transition-colors duration-300">
            <div className="w-full px-4 sm:px-6 lg:px-8">
                <div className="flex justify-between items-center h-20">

                    {/* Logo Section */}
                    <div className="flex items-center gap-3 cursor-pointer" onClick={() => navigate("/")}>
                        <div className="relative w-12 h-12 rounded-xl overflow-hidden shadow-sm">
                            <img
                                src="/ohglogo.png"
                                alt="OHG Logo"
                                className="w-full h-full object-cover"
                            />
                        </div>
                        <div className="flex flex-col justify-center gap-0.5">
                            <h1 className="text-xl font-extrabold leading-none tracking-tight">
                                <span className="text-ohg-navy-light">ONE</span>
                                <span className="text-ohg-orange mx-1">HUB</span>
                                <span className="text-ohg-teal">GLOBAL</span>
                            </h1>
                            <span className="text-[10px] text-muted-foreground font-semibold tracking-[0.2em] uppercase">One Hub Global</span>
                        </div>
                    </div>

                    {/* Desktop Navigation Middle */}
                    <div className="hidden lg:flex items-center space-x-8">
                        <Link to="/" className="text-sm font-semibold text-foreground hover:text-ohg-orange transition-colors border-b-2 border-ohg-navy dark:border-ohg-teal pb-1">
                            Home
                        </Link>

                        <Link to="/courses" className="text-sm font-medium text-muted-foreground hover:text-foreground transition-colors">
                            Courses
                        </Link>

                        <DropdownMenu>
                            <DropdownMenuTrigger className="flex items-center gap-1 text-sm font-medium text-muted-foreground hover:text-foreground transition-colors outline-none">
                                Menu <ChevronDown className="h-4 w-4" />
                            </DropdownMenuTrigger>
                            <DropdownMenuContent align="end" className="w-48 bg-card border-border">
                                <DropdownMenuItem>Option 1</DropdownMenuItem>
                                <DropdownMenuItem>Option 2</DropdownMenuItem>
                            </DropdownMenuContent>
                        </DropdownMenu>

                        <DropdownMenu>
                            <DropdownMenuTrigger className="flex items-center gap-1 text-sm font-medium text-muted-foreground hover:text-foreground transition-colors outline-none">
                                Tutorials <ChevronDown className="h-4 w-4" />
                            </DropdownMenuTrigger>
                            <DropdownMenuContent align="end" className="w-48 bg-card border-border">
                                <DropdownMenuItem>React</DropdownMenuItem>
                                <DropdownMenuItem>Python</DropdownMenuItem>
                            </DropdownMenuContent>
                        </DropdownMenu>

                        <Link to="/terminal" className="text-sm font-medium text-muted-foreground hover:text-foreground transition-colors">
                            Terminal
                        </Link>
                    </div>

                    {/* Desktop Right Actions */}
                    <div className="hidden lg:flex items-center gap-4">
                        <Button
                            className="bg-ohg-orange hover:bg-ohg-orange-hover text-white font-medium rounded-md px-6 shadow-md transition-all hover:shadow-lg dark:shadow-none"
                        >
                            Challenges
                        </Button>

                        <Button
                            variant="outline"
                            className="font-medium text-muted-foreground border-input hover:bg-accent hover:text-accent-foreground rounded-md px-6"
                        >
                            Apply Jobs
                        </Button>

                        <div className="w-px h-8 bg-border mx-2"></div>


                        <button className="p-2 text-muted-foreground hover:bg-accent hover:text-accent-foreground rounded-full transition-colors">
                            <Search className="h-5 w-5" />
                        </button>
                    </div>

                    {/* Mobile Menu Toggle */}
                    <div className="lg:hidden">
                        <Sheet>
                            <SheetTrigger asChild>
                                <Button variant="ghost" size="icon" className="text-foreground">
                                    <MenuIcon className="h-6 w-6" />
                                </Button>
                            </SheetTrigger>
                            <SheetContent side="right" className="bg-background border-border p-0">
                                <div className="p-6 border-b border-border flex items-center justify-between">
                                    <div className="flex items-center gap-3">
                                        <img src="/ohglogo.png" alt="Logo" className="w-8 h-8 rounded-lg" />
                                        <span className="font-bold text-foreground">OHG365</span>
                                    </div>

                                </div>
                                <div className="flex flex-col p-6 space-y-6">
                                    <Link to="/" className="text-lg font-medium text-foreground">Home</Link>
                                    <Link to="/courses" className="text-lg font-medium text-muted-foreground">Courses</Link>
                                    <Link to="/terminal" className="text-lg font-medium text-muted-foreground">Terminal</Link>
                                    <div className="h-px bg-border my-2"></div>
                                    <Button className="w-full bg-ohg-orange hover:bg-ohg-orange-hover text-white">Challenges</Button>
                                    <Button variant="outline" className="w-full">Apply Jobs</Button>
                                </div>
                            </SheetContent>
                        </Sheet>
                    </div>

                </div>
            </div>
        </nav>
    );
};
