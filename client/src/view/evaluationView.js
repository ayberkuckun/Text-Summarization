import '../App.css';
import {useState, useEffect} from 'react';
import {store} from '../util';

const EvaluationView = props => {
  const [metrics, setMetrics] = useState(null);
  const metricNames = ["rouge-1", "rouge-2", "rouge-l", "rouge-w"]

  useEffect(() => {
    store.set(setMetrics, "setMetrics");
  }, []);

  return (
    <div className="evaluation-view">
      {metrics && metricNames.map((item, i) => <div className="evaluation-item" key={i}>{item}: {metrics[0][item].toFixed(3)}</div>)}
    </div>
  );
}

export default EvaluationView;