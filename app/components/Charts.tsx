//@ts-ignore
import React from "react";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  RadialLinearScale,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";
import { Bar, Pie } from "react-chartjs-2";

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  ArcElement,
  RadialLinearScale,
  Title,
  Tooltip,
  Legend
);

type ChartsProps = {
  data: object;
};

function prepareAvgTimeChartData(obj: any): {
  labels: string[];
  data: number[];
} {
  const labels = obj.map((item: any) => item["name"]);
  const values = obj.map((item: any) => item["time to event"]["avg"]);

  return { labels, data: values };
}

function prepareCostChartData(obj: any): { data: number } {
  const values = obj["cost"]["avg"];

  return { data: values };
}

function prepareLYChartData(obj: any): { data: number } {
  const values = obj["LY"]["avg"];

  return { data: values };
}

function prepareQALYChartData(obj: any): { data: number } {
  const values = obj["QALY"]["avg"];

  return { data: values };
}

export default function Charts({ data }: ChartsProps) {
  const dataAVGTimeBase = prepareAvgTimeChartData(
    data["interventions"][0]["chronic manifestations"]
  );
  const dataAVGTimeObj = prepareAvgTimeChartData(
    data["interventions"][1]["chronic manifestations"]
  );

  const dataCostBase = prepareCostChartData(data["interventions"][0]);
  const dataCostObj = prepareCostChartData(data["interventions"][1]);

  const dataCost = [dataCostBase.data, dataCostObj.data];

  const dataLYBase = prepareLYChartData(data["interventions"][0]);
  const dataLYObj = prepareLYChartData(data["interventions"][1]);

  const dataLY = [dataLYBase.data, dataLYObj.data];

  const dataQALYBase = prepareQALYChartData(data["interventions"][0]);
  const dataQALYObj = prepareQALYChartData(data["interventions"][1]);

  const dataQALY = [dataQALYBase.data, dataQALYObj.data];

  const optionsAvgTime = {
    indexAxis: "y" as const,
    elements: {
      bar: {
        borderWidth: 2,
      },
    },
    responsive: true,
    plugins: {
      legend: {
        position: "right" as const,
      },
      title: {
        display: true,
        text: "Tiempo promedio hasta evento",
      },
    },
  };

  const optionsCost = {
    indexAxis: "y" as const,
    elements: {
      arc: {
        borderWidth: 2,
      },
    },
    responsive: true,
    plugins: {
      legend: {
        position: "bottom" as const,
      },
      title: {
        display: true,
        text: "Coste ($)",
      },
    },
  };

  const optionsLY = {
    indexAxis: "y" as const,
    elements: {
      arc: {
        borderWidth: 2,
      },
    },
    responsive: true,
    plugins: {
      legend: {
        position: "bottom" as const,
      },
      title: {
        display: true,
        text: "Esperanza de vida (años)",
      },
    },
  };

  const optionsQALY = {
    indexAxis: "y" as const,
    elements: {
      arc: {
        borderWidth: 2,
      },
    },
    responsive: true,
    plugins: {
      legend: {
        position: "bottom" as const,
      },
      title: {
        display: true,
        text: "Esperanza de vida con calidad (años)",
      },
    },
  };

  const chartDataAvgTime = {
    labels: dataAVGTimeBase.labels,
    datasets: [
      {
        label: "Tiempo promedio hasta evento (base)",
        data: dataAVGTimeBase.data,
        borderColor: "rgb(255, 99, 132)",
        backgroundColor: "rgba(255, 99, 132, 0.5)",
      },
      {
        label: "Tiempo promedio hasta evento (objetivo)",
        data: dataAVGTimeObj.data,
        borderColor: "rgb(54, 162, 235)",
        backgroundColor: "rgba(54, 162, 235, 0.2)",
      },
    ],
  };

  const chartDataQALY = {
    labels: ["Coste (base)", "Coste (objetivo)"],
    datasets: [
      {
        label: "Coste ($)",
        data: dataQALY,
        backgroundColor: ["rgb(255, 99, 132)", "rgb(54, 162, 235)"],
      },
    ],
  };

  const chartDataCost = {
    labels: ["Esperanza de vida (base)", "Esperanza de vida (objetivo)"],
    datasets: [
      {
        label: "Esperanza de vida (años)",
        data: dataCost,
        backgroundColor: ["rgb(255, 99, 132)", "rgb(54, 162, 235)"],
      },
    ],
  };

  const chartDataLY = {
    labels: [
      "Esperanza de vida con calidad (base)",
      "Esperanza de vida con calidad (objetivo)",
    ],
    datasets: [
      {
        label: "Esperanza de vida con calidad (años)",
        data: dataLY,
        backgroundColor: ["rgb(255, 99, 132)", "rgb(54, 162, 235)"],
      },
    ],
  };

  return (
    <div className="mx-auto max-w-7xl sm:px-6 lg:px-8">
      <Bar options={optionsAvgTime} data={chartDataAvgTime} />
      <div className="mb-4 flex flex-row" style={{ minHeight: "500px" }}>
        <div className="w-1/3">
          <Pie options={optionsCost} data={chartDataCost} />
        </div>
        <div className="w-1/3">
          <Pie options={optionsLY} data={chartDataLY} />
        </div>
        <div className="w-1/3">
          <Pie options={optionsQALY} data={chartDataQALY} />
        </div>
      </div>
    </div>
  );
}
