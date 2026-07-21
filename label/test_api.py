"""
This file contains tests for the API of your model. You can run these tests by installing test requirements:

    ```bash
    pip install -r requirements-test.txt
    ```
Then execute `pytest` in the directory of this file.

- Change `NewModel` to the name of the class in your model.py file.
- Change the `request` and `expected_response` variables to match the input and output of your model.
"""
import os.path

import pytest
import json
from model import SuryaOCR
import responses


@pytest.fixture
def client():
    from _wsgi import init_app
    app = init_app(model_class=SuryaOCR)
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def model_dir_env(tmp_path, monkeypatch):
    model_dir = tmp_path / "model_dir"
    model_dir.mkdir()
    monkeypatch.setattr(SuryaOCR, 'MODEL_DIR', str(model_dir))
    return model_dir


@responses.activate
def test_predict(client, model_dir_env):
    responses.add(
        responses.GET,
        'http://test_predict.surya.ml-backend.com/image.jpeg',
        body=open(os.path.join(os.path.dirname(__file__), 'test_images', 'image.jpeg'), 'rb').read(),
        status=200
    )
    request = {
        'tasks': [{
            'data': {
                'image': 'http://test_predict.surya.ml-backend.com/image.jpeg'
            }
        }],
        # Your labeling configuration here
        'label_config': '''
<View>
  <Image name="image" value="$image"/>

  <Labels name="label" toName="image">
    <Label value="Caption" background="#4E79A7"/>
    <Label value="Footnote" background="#F28E2B"/>
    <Label value="Equation" background="#E15759"/>
    <Label value="ListGroup" background="#76B7B2"/>
    <Label value="PageHeader" background="#59A14F"/>
    <Label value="PageFooter" background="#EDC948"/>
    <Label value="Picture" background="#B07AA1"/>
    <Label value="SectionHeader" background="#FF9DA7"/>
    <Label value="Table" background="#9C755F"/>
    <Label value="Text" background="#BAB0AC"/>
    <Label value="Figure" background="#1F77B4"/>
    <Label value="Code" background="#FF7F0E"/>
    <Label value="Form" background="#2CA02C"/>
    <Label value="TableOfContents" background="#D62728"/>
    <Label value="ChemicalBlock" background="#9467BD"/>
    <Label value="Diagram" background="#8C564B"/>
    <Label value="Bibliography" background="#E377C2"/>
    <Label value="BlankPage" background="#7F7F7F"/>
  </Labels>

  <Rectangle name="bbox" toName="image" strokeWidth="3"/>
  <Polygon name="poly" toName="image" strokeWidth="3"/>

  <TextArea name="transcription" toName="image"
            editable="true"
            perRegion="true"
            required="true"
            maxSubmissions="1"
            rows="5"
            placeholder="Recognized Text"
            displayMode="region-list"
            />
</View>
'''
    }

    response = client.post('/predict', data=json.dumps(request), content_type='application/json')
    assert response.status_code == 200
    response = json.loads(response.data)
    expected_texts = [
        '',
        '<p>IZIN SOLUTIONS</p>',
        '<h1>KENAPA<br/>HARUS<br/>PUNYA<br/>IMB?</h1>',
        '',
        '<p>Kenapa Harus Punya IMB?</p>',
        '<p>Swipe →</p>'
    ]
    texts_response = []
    labels_response = []
    with open(os.path.join(os.path.dirname(__file__), 'label_mappings.json'), 'r') as f:
        interface_labels = set(json.load(f).values())
    for r in response['results'][0]['result']:
        if r['from_name'] == 'label':
            assert r['type'] == 'labels'
            assert r['value']['labels'][0] in interface_labels
            labels_response.append(r['value']['labels'][0])
        if r['from_name'] == 'transcription':
            assert r['type'] == 'textarea'
            assert 'labels' not in r['value']
            texts_response.append(r['value']['text'][0])
    assert labels_response
    assert texts_response == expected_texts
