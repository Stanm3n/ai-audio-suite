// Exercise the real browser script with a small DOM/network fixture.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

class Element {
    constructor() {
        this.listeners = {};
        this.children = new Map();
        this.dataset = {};
        this.style = {};
        this.value = '';
        this.classList = { add() {}, remove() {} };
        this.parentNode = { replaceChild() {} };
    }
    addEventListener(name, callback) { this.listeners[name] = callback; }
    querySelector(selector) {
        if (selector.startsWith('input[name')) return null;
        if (!this.children.has(selector)) this.children.set(selector, new Element());
        return this.children.get(selector);
    }
    querySelectorAll() { return []; }
    cloneNode() { return this; }
    appendChild() {}
    prepend() {}
    click() {}
    remove() {}
}

async function setup() {
    const elements = new Map();
    const get = id => {
        if (!elements.has(id)) elements.set(id, new Element());
        return elements.get(id);
    };
    const card = new Element();
    get('job-card-template').content = { cloneNode: () => ({ querySelector: () => card }) };
    const calls = [];
    const intervals = new Map();
    const alerts = [];
    let ready, resultStatus = 200;
    const document = {
        addEventListener: (name, callback) => { ready = callback; },
        getElementById: get,
        querySelector: selector => selector.startsWith('.job-card[') ? card : new Element(),
        createElement: () => new Element(),
        body: new Element(),
    };
    const context = {
        document, console: { log() {}, error() {} },
        alert: message => alerts.push(message),
        Option: function(label, value) { this.value = value; },
        FormData: class {
            constructor() { this.values = new Map(); }
            set(key, value) { this.values.set(key, value); }
            get(key) { return this.values.get(key); }
            delete(key) { this.values.delete(key); }
        },
        setInterval: callback => { intervals.set(1, callback); return 1; },
        clearInterval: id => intervals.delete(id),
        window: { URL: { createObjectURL: () => 'blob:test', revokeObjectURL() {} } },
        fetch: async (url, options = {}) => {
            calls.push({ url, options });
            let status = 200, data = {};
            if (url === '/api/v1/system/status') {
                data = { supported_models: ['medium'], supported_translation_languages: ['de'], available_devices: ['cpu'] };
            } else if (url === '/api/v1/transcribe') {
                status = 202;
                data = { job_id: 'job-1', status: 'queued', access_token: 'private-token' };
            } else if (url.endsWith('/cancel')) {
                data = { cancelled: true };
            } else if (url === '/api/v1/results/job-1') {
                status = resultStatus;
                data = { status: 'complete', result: { segments: [{ start: 0, end: 1, text: 'hello' }] } };
            }
            return { status, ok: status < 400, json: async () => data, blob: async () => ({}), headers: { get: () => null } };
        },
    };
    vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../static/script.js'), 'utf8'), context);
    ready();
    await Promise.resolve();
    get('file-input').listeners.change({ target: { files: [{ name: 'audio.wav' }] } });
    await get('upload-form').listeners.submit({ preventDefault() {} });
    assert.deepEqual(alerts, []);
    return { card, calls, intervals, alerts, setResultStatus: status => { resultStatus = status; } };
}

test('upload token is sent in headers for polling, cancellation and download', async () => {
    const fixture = await setup();
    await fixture.card.querySelector('.cancel-btn').listeners.click();
    await fixture.intervals.get(1)();
    await fixture.card.querySelector('.download-srt').listeners.click();
    const protectedCalls = fixture.calls.filter(c => c.url.startsWith('/api/v1/results/') || c.url.endsWith('/cancel'));
    assert.equal(protectedCalls.length, 3);
    for (const call of protectedCalls) {
        assert.equal(call.options.headers.Authorization, 'Bearer private-token');
        assert.equal(call.url.includes('private-token'), false);
    }
    assert.deepEqual(fixture.alerts, []);
    assert.equal(fixture.intervals.size, 0);
});

test('polling stops when access is rejected', async () => {
    const fixture = await setup();
    fixture.setResultStatus(403);
    await fixture.intervals.get(1)();
    assert.equal(fixture.intervals.size, 0);
});
