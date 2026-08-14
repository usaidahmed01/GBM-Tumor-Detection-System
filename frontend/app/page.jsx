'use client';

import { useRouter } from 'next/navigation';
import { useState } from 'react';
import { motion, MotionConfig, useReducedMotion } from 'motion/react';

const UUID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

function Mark() {
  return (
    <div className="brand-mark" aria-hidden="true">
      <span className="brand-mark__ring" />
      <span className="brand-mark__cross brand-mark__cross--v" />
      <span className="brand-mark__cross brand-mark__cross--h" />
      <span className="brand-mark__core" />
    </div>
  );
}

export default function HomePage() {
  const router = useRouter();
  const reduceMotion = useReducedMotion();
  const [studyUuid, setStudyUuid] = useState('');
  const [error, setError] = useState('');

  function openStudy(event) {
    event.preventDefault();
    const normalized = studyUuid.trim();
    if (!UUID_PATTERN.test(normalized)) {
      setError('Enter a valid study UUID from a completed volumetric analysis.');
      return;
    }
    setError('');
    router.push(`/viewer/${normalized}`);
  }

  return (
    <MotionConfig reducedMotion="user">
      <main className="launch-shell">
        <div className="ambient ambient--one" />
        <div className="ambient ambient--two" />
        <motion.section
          className="launch-card glass-panel"
          initial={reduceMotion ? { opacity: 0 } : { opacity: 0, y: 18, scale: 0.985 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          transition={{ duration: 0.55, ease: [0.16, 1, 0.3, 1] }}
        >
          <div className="launch-card__brand">
            <Mark />
            <div>
              <div className="eyebrow">GBM CDSS · RESEARCH WORKSPACE</div>
              <h1>Clinical MRI Viewer</h1>
            </div>
          </div>

          <p className="launch-copy">
            Review a completed multimodal MRI analysis in synchronized orthographic views with
            AI segmentation overlays, quantitative measurements and atlas-derived localization.
          </p>

          <div className="launch-feature-row" aria-label="Viewer capabilities">
            <span>3-plane MPR</span>
            <span>WT / TC / ET</span>
            <span>Protected assets</span>
          </div>

          <form className="study-launch-form" onSubmit={openStudy}>
            <label htmlFor="studyUuid">Study UUID</label>
            <div className="study-launch-form__row">
              <input
                id="studyUuid"
                value={studyUuid}
                onChange={(event) => setStudyUuid(event.target.value)}
                placeholder="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
                autoComplete="off"
                spellCheck="false"
              />
              <motion.button
                type="submit"
                whileHover={reduceMotion ? undefined : { scale: 1.02 }}
                whileTap={reduceMotion ? undefined : { scale: 0.98 }}
              >
                Open viewer
                <span aria-hidden="true">→</span>
              </motion.button>
            </div>
            {error ? <div className="form-error" role="alert">{error}</div> : null}
          </form>

          <div className="research-notice">
            <span className="status-dot" />
            AI-assisted research prototype. Segmentation is not a definitive GBM diagnosis and
            clinician verification remains required.
          </div>
        </motion.section>
      </main>
    </MotionConfig>
  );
}
