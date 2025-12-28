
import { useState, useEffect, useCallback, useRef } from 'react';
import { useToast } from './use-toast';

interface UseExamProctoringProps {
    isActive: boolean;
    onBan?: () => void;
}

export const useExamProctoring = ({ isActive, onBan }: UseExamProctoringProps) => {
    const { toast } = useToast();
    const [isFullscreen, setIsFullscreen] = useState<boolean>(() => !!document.fullscreenElement); // Initialize with actual state
    const [tabSwitchCount, setTabSwitchCount] = useState<number>(0);
    const [isBanned, setIsBanned] = useState<boolean>(false);

    // Use a ref to debounce violations (prevent double counting blur + visibilitychange)
    const lastViolationTime = useRef<number>(0);

    // Initialize state from local storage
    useEffect(() => {
        const storedBan = localStorage.getItem('exam_is_banned');
        const storedCount = localStorage.getItem('exam_tab_switch_count');

        if (storedBan === 'true') {
            setIsBanned(true);
            if (onBan) onBan();
        }

        if (storedCount) {
            setTabSwitchCount(parseInt(storedCount, 10));
        }

        // Check fullscreen status on mount
        setIsFullscreen(!!document.fullscreenElement);
    }, [onBan]);

    // Function to handle counting violations
    const handleViolation = useCallback(() => {
        // Prevent duplicate firing within 1 second logic
        const now = Date.now();
        if (now - lastViolationTime.current < 1000) return;
        lastViolationTime.current = now;

        if (isBanned) return;

        // Use function update to get latest prev count
        setTabSwitchCount(prevCount => {
            const newCount = prevCount + 1;

            // Persist
            localStorage.setItem('exam_tab_switch_count', newCount.toString());

            if (newCount >= 3) {
                // Ban Logic
                setIsBanned(true);
                localStorage.setItem('exam_is_banned', 'true');

                // Show termination message (optional, usually handled by component UI)
                toast({
                    title: "Interview Terminated",
                    description: "Multiple tab switches detected. Your session has been locked.",
                    variant: "destructive",
                    duration: Infinity,
                });

                if (onBan) onBan();
            } else {
                // Warning Logic
                toast({
                    title: "⚠️ Warning: Examination Violation",
                    description: `Warning ${newCount}/2: Do not switch tabs or exit the window.`,
                    variant: "destructive", // Red for warning
                    duration: 5000,
                });
            }
            return newCount;
        });
    }, [isBanned, onBan, toast]);

    // Monitor Fullscreen changes
    useEffect(() => {
        if (!isActive || isBanned) return;

        const handleFullscreenChange = () => {
            const isFs = !!document.fullscreenElement;
            setIsFullscreen(isFs);

            // If user manually exits fullscreen (and not banned), show warning or force logic?
            // Requirement: "If the user exits fullscreen manually... Show a warning popup... Force re-entry"
            // We'll handle the "Popup" in the UI by checking `!isFullscreen`
            // But we can also check for repeated exits if needed. The requirement says:
            // "If the user exits fullscreen more than the allowed limit (see rule 3)"
            // So exiting fullscreen counts as a tab switch/violation?
            // Requirement 1 says: "If the user exits fullscreen more than the allowed limit... End interview".
            // This implies exiting fullscreen MIGHT increment the violation count or have its own limit.
            // Rule 3 is "Tab Switch Rules" with 2 warnings.
            // So I will treat Flightscreen Exit as a Violation as well?
            // "If the user exits fullscreen manually... Show a warning popup... If the user exits fullscreen more than the allowed limit... End"
            // It implies exiting fullscreen counts towards the limit (or has a parallel limit). 
            // I will treat it as a violation to keep it unified under "Tab/Window Focus" broadly, 
            // OR I can just enforce re-entry. 
            // Let's treat "Exit Fullscreen" as a generic "Distraction" -> Violation? 
            // Actually, usually "Exit Fullscreen" is an explicit action. 
            // Let's trigger `handleViolation()` on fullscreen exit.

            if (!isFs) {
                handleViolation();
            }
        };

        document.addEventListener('fullscreenchange', handleFullscreenChange);
        return () => document.removeEventListener('fullscreenchange', handleFullscreenChange);
    }, [isActive, isBanned, handleViolation]);

    // Monitor Visibility and Blur
    useEffect(() => {
        if (!isActive || isBanned) return;

        const onVisibilityChange = () => {
            if (document.hidden) {
                handleViolation();
            }
        };

        const onBlur = () => {
            // Blur can fire when clicking inside an iframe or some UI elements, 
            // so we must be careful. But for an exam kiosk mode, window blur is usually a violation.
            // We'll rely on the debounce in handleViolation to handle redundancy with visibilityHidden.
            handleViolation();
        };

        document.addEventListener('visibilitychange', onVisibilityChange);
        window.addEventListener('blur', onBlur);
        // window.addEventListener('focus', ...) - we don't need to count focus, just blur.

        return () => {
            document.removeEventListener('visibilitychange', onVisibilityChange);
            window.removeEventListener('blur', onBlur);
        };
    }, [isActive, isBanned, handleViolation]);

    const enterFullscreen = useCallback(async () => {
        try {
            await document.documentElement.requestFullscreen();
        } catch (err) {
            console.error("Error attempting to enable fullscreen:", err);
        }
    }, []);

    return {
        isFullscreen,
        isBanned,
        tabSwitchCount,
        enterFullscreen
    };
};
