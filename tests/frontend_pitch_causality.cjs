// Execute the actual TSX component with installed dependencies; no new test framework.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const Module = require('node:module');
const frontend = path.resolve(__dirname, '../frontend');
const localRequire = Module.createRequire(path.join(frontend, 'package.json'));
const ts = localRequire('typescript');
const React = localRequire('react');
const { renderToStaticMarkup } = localRequire('react-dom/server');
const source = path.join(frontend, 'components/pitch/Pitch.tsx');
const javascript = ts.transpileModule(fs.readFileSync(source, 'utf8'), {
  compilerOptions: { jsx: ts.JsxEmit.ReactJSX, module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
}).outputText;
const compiled = new Module(source, module);
compiled.filename = source;
compiled.paths = Module._nodeModulePaths(path.dirname(source));
compiled._compile(javascript, source);
const Pitch = compiled.exports.default;
const frames = [
  { t: 0, xy: [[20, 30]], b: [25, 34], p: null, obs: [true] },
  { t: 1, xy: [[100, 66]], b: [35, 40], p: null, obs: [true] },
];
const props = {
  t: 0.5, frames, meta: {
    player_order: ['A1'], players: [{ id: 'A1', team: 'A', name: 'Player', shirt: 1, position: 'MID' }],
    teams: { A: { color: '#ef4444' }, B: { color: '#64748b' } },
  },
  events: [], focusPlayers: [], focusTeam: null, overlay: null, selectedPlayer: null,
  showLines: false, onPlayerClick() {},
};
const render = causalPlayback => renderToStaticMarkup(React.createElement(Pitch, { ...props, causalPlayback }));
const live = render(true);
const replay = render(false);
assert.match(live, /translate\(20,30\)/);
assert.match(replay, /translate\(60,48\)/);
frames[1] = { ...frames[1], xy: [[5, 3]], b: [50, 1], p: 'B' };
assert.equal(render(true), live, 'Future coordinates/context must not change a current live rendering');
assert.notEqual(render(false), replay, 'Retrospective rendering intentionally uses the subsequent sample');
props.t = 1;
assert.match(render(true), /translate\(5,3\)/, 'A measurement becomes available at its actual timestamp');
props.t = 0.5;
props.meta.player_order = Array.from({ length: 10 }, (_, i) => `A${i + 1}`);
props.meta.players = props.meta.player_order.map((id, i) => ({ id, team: 'A', name: 'Player', shirt: i + 1, position: 'MID' }));
props.meta.teams.A.short = 'A';
frames[0] = { ...frames[0], xy: Array.from({ length: 10 }, (_, i) => [20 + i, 30 + i]), obs: Array(10).fill(true), sl: [0,0,0,0,1,1,1,2,2,2] };
frames[1] = { ...frames[1], xy: Array.from({ length: 10 }, (_, i) => [70 + i, 20 + i]) };
props.shape = { A: { name: '4-2-3-1', lines: [props.meta.player_order] } };
const causalShape = render(true);
assert.match(causalShape, />A 4-3-3<\/text>/);
assert.doesNotMatch(causalShape, /4-2-3-1/);
props.shape.A.name = '2-2-6';
props.shape.A.lines = [];
assert.equal(render(true), causalShape, 'Retrospective/truth formation names must not change a causal pitch');
delete frames[0].sl;
assert.doesNotMatch(render(true), /A 2-2-6/, 'No retrospective line-group fallback in causal rendering');
console.log('Causal pitch rendering: future mutation invariance and retrospective interpolation passed');
// Video display must not relabel a future estimate as a current observation.
props.meta.source = { type: 'computer_vision' };
props.meta.player_order = ['A1'];
props.meta.players = [{ id: 'A1', team: 'A', name: 'Player', shirt: 1, position: 'MID' }];
props.shape = null;
frames[0] = { t: 0, xy: [[20,30]], obs: [true], b: [25,34], p: null };
frames[1] = { t: 1, xy: [[25,30]], obs: [false], b: [30,34], p: null };
assert.match(render(false), /translate\(20,30\)/);
frames[1].obs = [true];
assert.match(render(false), /translate\(22.5,30\)/, 'Smooth same-status video sightings can interpolate');
frames[1].xy = [[100,60]];
assert.match(render(false), /translate\(20,30\)/, 'Large identity/calibration jump is not a fabricated run');
frames[0].ds = 'Penalty';
frames[0].ds_basis = 'annotation_interval';
const stopped = render(false);
assert(stopped.includes('PENALTY · ANNOTATED STOPPAGE'));
assert(stopped.includes('cx="25" cy="34" r="0.75"'), 'Dead ball does not fly towards next restart sample');
assert(!stopped.includes('PENALTY · NOT TAKEN YET'));
console.log('Video discontinuity, honest stoppage labels and stationary dead-ball rendering passed');
