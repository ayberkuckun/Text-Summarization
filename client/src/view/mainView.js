import '../App.css';
import Header from './header';
import InputView from './inputView';
import EvaluationView from './evaluationView';
import OutputView from './outputView';

const MainView = ({methods, datasets}) => {

  return (
    <div className="main-view">
      <Header />
      <InputView methods={methods} datasets={datasets} />
      <EvaluationView />
      <OutputView />
    </div>
  );
}

export default MainView;