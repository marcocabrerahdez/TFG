import { useState } from "react";
import Header from "../components/Header";
import Form from "../components/Form";
import Charts from "../components/Charts";
import Loading from "@/components/Loading";
import Footer from "../components/Footer";

export default function Main() {
  const [showCharts, setShowCharts] = useState(false);
  const [data, setData] = useState({});

  const handleFormSubmit = (data: object) => {
    setShowCharts(true);
    setData(data);
  };

  return (
    <main className="flex flex-col items-center justify-between" style={{ backgroundColor: '#A8DADC'}}>
      <Header />
      <div className="grid grid-cols-2">
        <div className="col-span-1" style={{ padding: '1rem'}}>
          <Form onFormSubmit={handleFormSubmit} />
        </div>
        {
          showCharts ?
            <div className="col-span-1" style={{ padding: '1rem'}}>
              <Charts data={data} />
            </div>
            :
            <Loading />
        }
      </div>
      <Footer />
    </main>
  );
}
