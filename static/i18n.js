// ---- Localization ----
// All user-facing UI copy lives here, in both languages, keyed the same way
// in each. app.js's t(key, ...args) looks a key up in TRANSLATIONS[currentLanguage],
// falling back to English if a key is ever missing for the active language.
// A value can be a plain string, or a function for copy that needs to embed
// a runtime value (a count, a percentage, another already-translated string) -
// t() calls it with whatever args it was given past the key.
//
// Scope: this only covers this file's own UI chrome - headings, buttons,
// hints, status text. It does NOT cover class names (MNIST digits "0".."9",
// or Quick Draw category words like "umbrella") - those come from the
// server as data, not UI copy, and Quick Draw's own category CSV already
// has a "swedish" column reserved for exactly that, unused so far (see
// server/quickdraw_categories.py) - translating those would mean threading
// a parallel display name through the server's class_names wire format,
// which is a separate, bigger change from "translate the UI chrome."
//
// Loaded via a <script> tag before app.js - classic (non-module) scripts on
// the same page share one global scope, so TRANSLATIONS here is directly
// visible to app.js without any export/import machinery.
const TRANSLATIONS = {
  en: {
    pageTitle: "Train Your Own AI",
    menuSubtitle: "What should it learn to recognize?",
    digitsMode: "Digits",
    drawingsMode: "Drawings",
    loading: "Loading...",
    backToMenu: "Back to menu",
    drawHeadingDigits: "Draw a digit",
    drawHeadingDrawings: "Draw an object",
    reset: "Reset",
    configureHeading: "Configure the AI",
    addLayer: "+ Add",
    removeLayer: "− Remove",
    configHint: "Click a node to see what makes it fire",
    classificationHeading: "Classification",
    explainHintDigits: "Click a digit to see what ink would help or hurt it",
    explainHintDrawings: "Click a class to see what ink would help or hurt it",
    aiStatusHeading: "AI Status",
    // Four certainty tiers derived from how much more probable the top
    // prediction is than the runner-up (see app.js's certaintyLevel()) -
    // Unknown < Uncertain < Confident < Certain. Unknown/Uncertain's copy
    // never names the mode's noun, so those two keys aren't mode-suffixed;
    // Confident/Certain are, same as explainHint*/loadSample* above.
    certaintyUnknown: (label) => `The AI can't tell. Is it: ${label}?`,
    certaintyUncertain: (label) => `The AI is uncertain, but thinks it is: ${label}.`,
    certaintyConfidentDigits: (label) => `The AI is fairly confident the digit is: ${label}`,
    certaintyConfidentDrawings: (label) => `The AI is fairly confident the motif is: ${label}`,
    certaintyCertainDigits: (label) => `The AI is certain the digit is: ${label}`,
    certaintyCertainDrawings: (label) => `The AI is certain the motif is: ${label}`,
    train: "Train",
    lossLabel: "Loss",
    validationAccuracyLabel: "Validation accuracy",
    debugHeading: "More...",
    loadSampleDigits: "Load random test digit",
    loadSampleDrawings: "Load random test drawing",
    debugToggle: "More...",
    trainingEllipsis: "Training...",
    trainingComplete: "Training complete.",
    trainingError: "Training error.",
    idle: "idle",
    noTrainedModel: "No trained AI yet - press Train",
    epochsTrained: (n) => `epochs trained: ${n}`,
    valAccuracy: (pct) => `val accuracy: ${pct}%`,
    datasetSizes: (train, val) => `train/val samples: ${train}/${val}`,
    stoppingCondition: (reason) => `stopping condition: ${reason}`,
    stopReasonEarlyStopping: "early stopping (validation loss stopped improving)",
    stopReasonMaxEpochs: "reached max epochs",
    stopReasonStoppedByUser: "stopped (new run started or mode changed)",
    trueLabel: (label) => `True label: ${label} (compare to the prediction on the right)`,
    inactivityHeading: "Are you still there?",
    inactivityMessage: (seconds) => `The kiosk will reset in ${seconds}s due to inactivity.`,
    inactivityCancel: "Cancel",
    helpButtonLabel: "Help",
    helpClose: "Close",
  },
  sv: {
    pageTitle: "Träna din egen AI",
    menuSubtitle: "Vad ska den lära sig känna igen?",
    digitsMode: "Siffror",
    drawingsMode: "Teckningar",
    loading: "Laddar...",
    backToMenu: "Tillbaka till menyn",
    drawHeadingDigits: "Rita en siffra",
    drawHeadingDrawings: "Rita något",
    reset: "Rensa",
    configureHeading: "Konfigurera AI:n",
    addLayer: "+ Lägg till",
    removeLayer: "− Ta bort",
    configHint: "Klicka på en nod för att se vad som får den att aktiveras",
    classificationHeading: "Klassificering",
    explainHintDigits: "Klicka på en siffra för att se vilket område som skulle hjälpa eller stjälpa den",
    explainHintDrawings: "Klicka på en klass för att se vilket område som skulle hjälpa eller stjälpa den",
    aiStatusHeading: "AI-status",
    certaintyUnknown: (label) => `AI:n kan inte avgöra det. Är det: ${label}?`,
    certaintyUncertain: (label) => `AI:n är osäker, men tror att det är: ${label}.`,
    certaintyConfidentDigits: (label) => `AI:n är ganska säker på att siffran är: ${label}`,
    certaintyConfidentDrawings: (label) => `AI:n är ganska säker på att motivet är: ${label}`,
    certaintyCertainDigits: (label) => `AI:n är säker på att siffran är: ${label}`,
    certaintyCertainDrawings: (label) => `AI:n är säker på att motivet är: ${label}`,
    train: "Träna",
    lossLabel: "Förlust",
    validationAccuracyLabel: "Valideringsnoggrannhet",
    debugHeading: "Mer...",
    loadSampleDigits: "Ladda slumpmässig testsiffra",
    loadSampleDrawings: "Ladda slumpmässig testteckning",
    debugToggle: "Mer...",
    trainingEllipsis: "Tränar...",
    trainingComplete: "Träning klar.",
    trainingError: "Träningsfel.",
    idle: "inaktiv",
    noTrainedModel: "Ingen tränad AI än - tryck på Träna",
    epochsTrained: (n) => `tränade epoker: ${n}`,
    valAccuracy: (pct) => `valideringsnoggrannhet: ${pct}%`,
    datasetSizes: (train, val) => `tränings-/valideringsdata: ${train}/${val}`,
    stoppingCondition: (reason) => `stoppvillkor: ${reason}`,
    stopReasonEarlyStopping: "tidigt stopp (valideringsförlusten slutade förbättras)",
    stopReasonMaxEpochs: "nådde max antal epoker",
    stopReasonStoppedByUser: "stoppad (ny körning startad eller läge ändrat)",
    trueLabel: (label) => `Rätt svar: ${label} (jämför med förutsägelsen till höger)`,
    inactivityHeading: "Är du kvar?",
    inactivityMessage: (seconds) => `Kiosken återställs om ${seconds}s på grund av inaktivitet.`,
    inactivityCancel: "Avbryt",
    helpButtonLabel: "Hjälp",
    helpClose: "Stäng",
  },
};

// Every language must define exactly the same keys - catches a missing
// translation at load time (loudly, in the console) instead of silently
// falling back mid-demo. English is the reference set.
(function checkTranslationKeysMatch() {
  const referenceKeys = Object.keys(TRANSLATIONS.en).sort();
  for (const lang of Object.keys(TRANSLATIONS)) {
    if (lang === "en") continue;
    const keys = Object.keys(TRANSLATIONS[lang]).sort();
    const missing = referenceKeys.filter((k) => !keys.includes(k));
    const extra = keys.filter((k) => !referenceKeys.includes(k));
    if (missing.length || extra.length) {
      console.error(`TRANSLATIONS.${lang} is out of sync with TRANSLATIONS.en - missing: ${missing}, extra: ${extra}`);
    }
  }
})();
