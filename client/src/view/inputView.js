import '../App.css';
import {useState, useEffect} from 'react';
import {store, getReadableName} from '../util';

const InputView = props => {
  const [methodList, setMethodList] = useState([]);
  const [datasetList, setDatasetList] = useState([]);
  const [currentMethod, setCurrentMethod] = useState(0);
  const [inputView, setInputView] = useState(true);

  useEffect(() => {
    if (props.methods) {
      let arr = [];

      for (let i = 0; i < props.methods.length; i++) {
        arr.push(getReadableName(props.methods[i]));
      }

      setMethodList(arr);
    }
  }, [props.methods]);

  useEffect(() => {
    if (props.datasets) {
      let arr = [];

      for (let i = 0; i < props.datasets.length; i++) {
        arr.push(getReadableName(props.datasets[i]));
      }

      setDatasetList(arr);
    }
  }, [props.datasets]);

  const setSummarizeMethod = el => {
    const index = Number(el.target.value);
    setCurrentMethod(index);
  }

  const submit = ev => {
    ev.preventDefault();

    if (currentMethod == null) {
      return;
    }

    if (inputView) {
      const data = {method: currentMethod, text: ev.target.parentElement.parentElement.children[0].children[0].value, summary: ev.target.parentElement.parentElement.children[1].children[0].value};
      const options = {
              method: "POST",
              body: JSON.stringify({data}),
              headers: {"Content-Type" : "application/json"}
      };

      fetch("http://localhost:1234", options).then(res => {
        if (res.status === 200) {
          res.json().then(data => {
            store.get("setMetrics")(data.metrics);
            store.get("setOutput")(data.output);
          });
        }
      });
    } else {
      const value = ev.target.parentElement.parentElement.children[2].children[0].selectedIndex;
      const data = {dataset_num: Number(value)}
      
      const options = {
        method: "POST",
        body: JSON.stringify({data}),
        headers: {"Content-Type" : "application/json"}
      };

      fetch("http://localhost:1234/generate", options).then(res => {
        if (res.status === 200) {
          res.json().then(data => {
            store.get("setMetrics")(data.metrics);
            store.get("setOutput")(data.output);
          });
        }
      });
    }
  }

  const toggleMode = el => {
    if (inputView) {
      store.textValue = el.target.parentElement.parentElement.children[2].children[0].children[0].value;
      el.target.parentElement.parentElement.children[2].children[0].children[0].value = "";
      store.summaryValue = el.target.parentElement.parentElement.children[2].children[1].children[0].value;
      el.target.parentElement.parentElement.children[2].children[1].children[0].value = "";
      setInputView(false);
    } else {
      el.target.parentElement.parentElement.children[2].children[0].children[0].value = store.textValue;
      el.target.parentElement.parentElement.children[2].children[1].children[0].value = store.summaryValue;
      setInputView(true);
    }
  }

  return (
    <div className="input-view">
      <h2>Textual input</h2>
      <div className="input-control">
        <button className={inputView ? "input-control-button input-control-button-active" : "input-control-button"} onClick={toggleMode} value="input">INPUT</button>
        <button className={inputView ? "input-control-button" : "input-control-button input-control-button-active"} onClick={toggleMode} value="generate">GENERATION</button>
      </div>
      <form>
        <div className="input-item">
          < textarea className={inputView ? "" : "input-disabled"} disabled={!inputView}></textarea>
        </div>
        <div className="input-item">
          <input className={inputView ? "input-summary" : "input-summary input-disabled"} type="text" placeholder="Potential summary" disabled={!inputView} />
        </div>
        <div className="input-item">
        {inputView && <select className="option-select" onChange={setSummarizeMethod}>{methodList.length && methodList.map((item, i) => <option className="option-method" value={i} key={i}>{item}</option>)}</select>}
        {!inputView && <select className="option-select" onChange={setSummarizeMethod}>{methodList.length && datasetList.map((item, i) => <option className="option-method" value={i} key={i}>{item}</option>)}</select>}
        </div>
        <div className="input-item">
          <button type="submit" onClick={submit}>{inputView ?  "Summarize input" : "Generate summary example"}</button>
        </div>
      </form>
    </div>
  );
}

export default InputView;