"""Explicit request/response mapping for vLLM and native Ollama chat APIs."""


def inspect_model(session, api, model, backend):
    if backend == 'ollama':
        base = api.removesuffix('/chat')
        response = session.get(base+'/tags', timeout=15); response.raise_for_status()
        entry = next((m for m in response.json().get('models', []) if m.get('name') == model), None)
        if entry is None:
            raise ValueError('requested Ollama model is not installed')
        response = session.get(base+'/version', timeout=15); response.raise_for_status()
        return {'backend': backend, 'version': response.json(), 'model': entry}
    response = session.get(api.rsplit('/chat/completions', 1)[0]+'/models', timeout=15)
    response.raise_for_status()
    entry = next((m for m in response.json().get('data', []) if m.get('id') == model), None)
    if entry is None:
        raise ValueError('requested model is not advertised by this server')
    return {'backend': backend, 'model': entry}


def request_body(model, messages, max_tokens, temperature, seed, backend, num_ctx=4096):
    if backend == 'ollama':
        return {'model': model, 'messages': messages, 'stream': False, 'think': False,
                'keep_alive': '30m', 'options': {'temperature': temperature, 'seed': seed,
                'num_predict': max_tokens, 'num_ctx': num_ctx, 'top_p': 1., 'top_k': 0,
                'min_p': 0., 'repeat_penalty': 1., 'presence_penalty': 0.}}
    return {'model': model, 'messages': messages, 'max_tokens': max_tokens,
            'temperature': temperature, 'seed': seed,
            'chat_template_kwargs': {'enable_thinking': False}}


def response_content(data, backend, allow_empty=False):
    if backend == 'ollama':
        if data.get('done') is not True:
            raise ValueError('incomplete Ollama response')
        content = data['message']['content']
    else:
        content = data['choices'][0]['message']['content']
    if not isinstance(content, str):
        raise ValueError('non-text model output')
    if not content.strip() and allow_empty:
        reason = data.get('done_reason') if backend == 'ollama' else data['choices'][0].get('finish_reason')
        if reason not in ('stop', 'length'):
            raise ValueError('empty output without a completed generation')
        return ''
    if not content.strip():
        raise ValueError('empty/non-text model output')
    return content.strip()
