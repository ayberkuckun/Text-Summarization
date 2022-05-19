import '../App.css';
import {useState, useEffect} from 'react';
import {store} from '../util';

const OutputView = props => {
  const [output, setOutput] = useState(null);

  useEffect(() => {
    store.set(setOutput, "setOutput");
  }, []);

  return (
    <div className="output-view">
      <div className="output-item">
        {output && !output.results && <span className="output-text">{output}</span>}
        {output && output.results && <div className="output-item">
          <div className="output-item"><strong>Original text: </strong>{output.originalText}</div>
          <div className="output-item"><strong>Original summary: </strong>{output.originalSummary}</div>
          <strong>Predicted summary: </strong>{output.results.map((item, i) => <span className="output-text" key={i}>{item.summary}</span>)}</div>}
      </div>
    </div>
  );
}

export default OutputView;