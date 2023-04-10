import { useState } from "react";
import Header from "../components/Header";
import Form from "../components/Form";
import Charts from "../components/Charts";
import Footer from "../components/Footer";

export default function Main() {
  const [showCharts, setShowCharts] = useState(false);
  const [showForm, setShowForm] = useState(true);
  const [data, setData] = useState({});

  const handleFormSubmit = (data: object) => {
    setShowCharts(true);
    setShowForm(false);
    setData(data);
  };

  return (
    <main className="flex flex-col items-center justify-between">
      <Header />
      {showForm && <Form onFormSubmit={handleFormSubmit} />}
      {showCharts && <Charts data={data} />}
      <Footer />
    </main>
  );
}
