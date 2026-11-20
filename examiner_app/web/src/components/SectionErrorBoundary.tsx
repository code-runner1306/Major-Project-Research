import { Component, type ErrorInfo, type ReactNode } from 'react';
import './SectionErrorBoundary.css';

interface Props {
  children: ReactNode;
  /** Replaces the default "This part failed to load" box. */
  fallback?: (error: Error) => ReactNode;
}

interface State {
  error: Error | null;
}

/**
 * Catches a render error in one part of a page so the rest of the page keeps working.
 * Shows a small box with the error message by default.
 */
export class SectionErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error('SectionErrorBoundary caught an error', error, info.componentStack);
  }

  render(): ReactNode {
    const { error } = this.state;
    if (error === null) return this.props.children;
    if (this.props.fallback) return this.props.fallback(error);
    return (
      <div className="section-error" role="alert">
        <p>
          This part failed to load: <strong>{error.message || 'unknown error'}</strong>. Other sections still work.
        </p>
      </div>
    );
  }
}
