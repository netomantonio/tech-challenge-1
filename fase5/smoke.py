"""Verifica servidor HTTP real e análise demo dentro do container."""
import json
import subprocess
import sys
import time
import urllib.request

from fase5.train import OUTPUT


def main():
    process = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'fase5.app:app', '--port', '8051'])
    try:
        for _ in range(50):
            try:
                with urllib.request.urlopen('http://127.0.0.1:8051/health', timeout=1) as response:
                    assert response.status == 200
                break
            except OSError:
                if process.poll() is not None:
                    raise RuntimeError('Servidor encerrou antes de iniciar.')
                time.sleep(.2)
        else:
            raise RuntimeError('Servidor não iniciou no prazo.')
        with urllib.request.urlopen('http://127.0.0.1:8051/', timeout=5) as response:
            assert response.status == 200 and 'Guardiã AI' in response.read().decode('utf-8')
        entry = json.loads((OUTPUT / 'exemplos.json').read_text(encoding='utf-8'))[0]['entrada']
        request = urllib.request.Request('http://127.0.0.1:8051/analisar',
                                         data=json.dumps(entry).encode(),
                                         headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=20) as response:
            result = json.load(response)
        assert result['auditoria_registrada']
        assert result['provedor_efetivo'] == 'demo-sem-llm'
        assert result['revisao_humana_obrigatoria']
        print(json.dumps({'http': 200, 'interface_web': 200, 'auditoria': True, 'modo': 'demo-sem-llm'}))
    finally:
        process.terminate()
        process.wait(timeout=10)


if __name__ == '__main__':
    main()
