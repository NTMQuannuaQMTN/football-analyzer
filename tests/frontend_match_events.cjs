// Render the actual event overlay/timeline with existing TypeScript and React.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const Module = require('node:module');
const frontend = path.resolve(__dirname, '../frontend');
const localRequire = Module.createRequire(path.join(frontend, 'package.json'));
const ts = localRequire('typescript');
const React = localRequire('react');
const { renderToStaticMarkup } = localRequire('react-dom/server');
const resolve = Module._resolveFilename;
Module._resolveFilename = function(request, parent, ...args) {
  return resolve.call(this, request.startsWith('@/') ? path.join(frontend, request.slice(2)) : request, parent, ...args);
};
for (const extension of ['.ts', '.tsx']) Module._extensions[extension] = (module, filename) => {
  module._compile(ts.transpileModule(fs.readFileSync(filename, 'utf8'), {
    compilerOptions: { jsx: ts.JsxEmit.ReactJSX, module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
  }).outputText, filename);
};
const MatchEvents = require(path.join(frontend, 'components/studio/MatchEvents.tsx')).default;
const MatchStory = require(path.join(frontend, 'components/timeline/MatchStory.tsx')).default;
const data = JSON.parse(fs.readFileSync(path.resolve(__dirname, '../data/cv_2016_08_14_18_00_arsenal_3_4_liverpool_h1_1440_2040.json')));
const meta = { ...data.meta, players: data.players };
const noop = () => {};
const html = renderToStaticMarkup(React.createElement(MatchEvents, { events: data.events, meta, onSeek: noop, onClose: noop }));
assert.equal((html.match(/<table\b/g) || []).length, 1);
assert.equal((html.match(/Jump to [^"]*free-kick at/g) || []).length, 5);
assert(html.includes('Jump to Penalty at 29:02'));
assert(html.includes('SoccerNet annotation'));
assert(html.includes('does not establish its outcome or taker'));
assert(!html.includes('<pre'));
assert(!html.includes('undefined'));
const timeline = { A: [], B: [] };
const story = renderToStaticMarkup(React.createElement(MatchStory, {
  events: data.events, meta, story: { chapters: [], edges: [] }, moments: [], timeline,
  t: 0, selectedId: null, onSelect: noop, onSeek: noop,
}));
assert.equal((story.match(/Jump to /g) || []).length, 6);
assert(story.includes('Jump to Penalty at 29:02'));
assert(!story.includes('Jump to Corner')); // No corner annotation exists in this clip.
const empty = renderToStaticMarkup(React.createElement(MatchEvents, { events: [], meta, onSeek: noop, onClose: noop }));
assert(empty.includes('No recorded events in the available window'));
console.log('Actual event overlay/timeline: five source free kicks, penalty clock offset, native table, no invented corner/outcome passed');
