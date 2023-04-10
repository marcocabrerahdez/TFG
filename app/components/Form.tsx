import { useState } from "react";

type FormProps = {
  onFormSubmit: (data: object) => void;
};

export default function Form({ onFormSubmit }: FormProps) {
  const [man, setMan] = useState("");
  const [age, setAge] = useState("");
  const [durationOfDiabetes, setDurationOfDiabetes] = useState("");
  const [baseHbA1cLevel, setBaseHbA1cLevel] = useState("");
  const [objHbA1cLevel, setObjHbA1cLevel] = useState("");
  const [hypoRate, setHypoRate] = useState("");
  const [hypoRateRR, setHypoRateRR] = useState("");
  const [annualCost, setAnnualCost] = useState("");
  const [manifestations, setManifestations] = useState<string[]>([]);

  const handleManifestationsChange = (event: {
    target: { value: string; checked: boolean };
  }) => {
    const value = event.target.value;
    if (event.target.checked) {
      setManifestations([...manifestations, value]);
    } else {
      setManifestations(manifestations.filter((m) => m !== value));
    }
  };
  console.log(manifestations);

  const handleSubmit = (event: { preventDefault: () => void }) => {
    event.preventDefault();
    // Aquí se puede agregar la lógica para enviar los datos del formulario a un servidor o realizar alguna acción adicional
    const formData = {
      man,
      age,
      durationOfDiabetes,
      hypoRate,
      hypoRateRR,
      baseHbA1cLevel,
      objHbA1cLevel,
      annualCost,
      manifestations,
    };

    const url = "http://127.0.0.1:5000/diabetes";
    const options = {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
      },
      body: JSON.stringify(formData),
    };

    fetch(url, options)
      .then((response) => response.json())
      .then((data) => {
        onFormSubmit(data);
      })
      .catch((error) => console.error(error));
  };

  return (
    <form onSubmit={handleSubmit} className="max-w-8xl my-auto">
      <div className="grid grid-cols-3 gap-4">
        <div className="mb-4">
          <label htmlFor="man" className="block mb-2 font-bold text-gray-600">
            Sexo
          </label>
          <select
            id="man"
            name="man"
            value={man}
            onChange={(event) => setMan(event.target.value)}
            className="border border-gray-300 p-2 w-full"
          >
            <option value="">Seleccione una opción</option>
            <option value="true">Masculino</option>
            <option value="false">Femenino</option>
          </select>
        </div>
        <div className="mb-4">
          <label htmlFor="age" className="block mb-2 font-bold text-gray-600">
            Edad
          </label>
          <input
            type="number"
            id="age"
            name="age"
            value={age}
            placeholder="Introduzca un número"
            onChange={(event) => setAge(event.target.value)}
            className="border border-gray-300 p-2 w-full"
          />
        </div>
        <div className="mb-4">
          <label
            htmlFor="durationOfDiabetes"
            className="block mb-2 font-bold text-gray-600"
          >
            Duración de la diabetes (en años)
          </label>
          <input
            type="number"
            id="durationOfDiabetes"
            name="durationOfDiabetes"
            value={durationOfDiabetes}
            placeholder="Introduzca un número"
            onChange={(event) => setDurationOfDiabetes(event.target.value)}
            className="border border-gray-300 p-2 w-full"
          />
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="mb-4">
          <label
            htmlFor="baseHbA1cLevel"
            className="block mb-2 font-bold text-gray-600"
          >
            Nivel Base de HbA1c (%)
          </label>
          <input
            type="number"
            id="baseHbA1cLevel"
            name="baseHbA1cLevel"
            value={baseHbA1cLevel}
            placeholder="Introduzca un número"
            onChange={(event) => setBaseHbA1cLevel(event.target.value)}
            className="border border-gray-300 p-2 w-full"
          />
        </div>
        <div className="mb-4">
          <label
            htmlFor="objHbA1cLevel"
            className="block mb-2 font-bold text-gray-600"
          >
            Nivel Objetivo de HbA1c (%)
          </label>
          <input
            type="number"
            id="objHbA1cLevel"
            name="objHbA1cLevel"
            value={objHbA1cLevel}
            placeholder="Introduzca un número"
            onChange={(event) => setObjHbA1cLevel(event.target.value)}
            className="border border-gray-300 p-2 w-full"
          />
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="mb-4">
          <label
            htmlFor="hypoRate"
            className="block mb-2 font-bold text-gray-600"
          >
            Episodios de hipoglucémicos severos (al año)
          </label>
          <input
            type="number"
            id="hypoRate"
            name="hypoRate"
            value={hypoRate}
            placeholder="Introduzca un número"
            onChange={(event) => setHypoRate(event.target.value)}
            className="border border-gray-300 p-2 w-full"
          />
        </div>
        <div className="mb-4">
          <label
            htmlFor="hypoRateRR"
            className="block mb-2 font-bold text-gray-600"
          >
            Objetivo de episodios de hipoglucémicos severos (al año)
          </label>
          <input
            type="number"
            id="hypoRateRR"
            name="hypoRateRR"
            placeholder="Introduzca un número"
            value={hypoRateRR}
            onChange={(event) => setHypoRateRR(event.target.value)}
            className="border border-gray-300 p-2 w-full"
          />
        </div>
      </div>
      <div className="mb-4">
        <label
          htmlFor="annualCost"
          className="block mb-2 font-bold text-gray-600"
        >
          Costo de tratamiento (en dólares)
        </label>
        <input
          type="number"
          id="annualCost"
          name="annualCost"
          value={annualCost}
          placeholder="Introduzca un número"
          onChange={(event) => setAnnualCost(event.target.value)}
          className="border border-gray-300 p-2 w-full"
        />
      </div>
      <div className="mb-4">
        <label
          htmlFor="manifestations"
          className="block mb-2 font-bold text-gray-600"
        >
          Manifestaciones clínicas
        </label>
        <div className="grid grid-cols-2 gap-4">
          <label
            htmlFor="Retinopatía de Fondo"
            className="inline-flex items-center mt-2"
          >
            <input
              id="BGRET"
              name="manifestations"
              type="checkbox"
              value="BGRET"
              onChange={handleManifestationsChange}
              checked={manifestations.includes("BGRET")}
              className="form-checkbox h-5 w-5 text-gray-600"
            />
            <span className="ml-2 text-gray-600">Retinopatía de Fondo</span>
          </label>
          <label
            htmlFor="Retinopatía Proliferativa"
            className="inline-flex items-center mt-2"
          >
            <input
              id="PRET"
              name="manifestations"
              type="checkbox"
              value="PRET"
              onChange={handleManifestationsChange}
              checked={manifestations.includes("PRET")}
              className="form-checkbox h-5 w-5 text-gray-600"
            />
            <span className="ml-2 text-gray-600">
              Retinopatía Proliferativa
            </span>
          </label>
          <label
            htmlFor="Edema Macular Diabético"
            className="inline-flex items-center mt-2"
          >
            <input
              id="ME"
              name="manifestations"
              type="checkbox"
              value="ME"
              onChange={handleManifestationsChange}
              checked={manifestations.includes("ME")}
              className="form-checkbox h-5 w-5 text-gray-600"
            />
            <span className="ml-2 text-gray-600">Edema Macular Diabético</span>
          </label>
          <label
            htmlFor="Enfermedad Renal Terminal"
            className="inline-flex items-center mt-2"
          >
            <input
              id="ESRD"
              name="manifestations"
              type="checkbox"
              value="ESRD"
              onChange={handleManifestationsChange}
              checked={manifestations.includes("ESRD")}
              className="form-checkbox h-5 w-5 text-gray-600"
            />
            <span className="ml-2 text-gray-600">
              Enfermedad Renal Terminal
            </span>
          </label>
          <label htmlFor="Ceguera" className="inline-flex items-center mt-2">
            <input
              id="BLI"
              name="manifestations"
              type="checkbox"
              value="BLI"
              onChange={handleManifestationsChange}
              checked={manifestations.includes("BLI")}
              className="form-checkbox h-5 w-5 text-gray-600"
            />
            <span className="ml-2 text-gray-600">Ceguera</span>
          </label>
          <label
            htmlFor="Microalbuminuria"
            className="inline-flex items-center mt-2"
          >
            <input
              id="ALB1"
              name="manifestations"
              type="checkbox"
              value="ALB1"
              onChange={handleManifestationsChange}
              checked={manifestations.includes("ALB1")}
              className="form-checkbox h-5 w-5 text-gray-600"
            />
            <span className="ml-2 text-gray-600">Microalbuminuria</span>
          </label>
          <label
            htmlFor="Macroalbuminuria"
            className="inline-flex items-center mt-2"
          >
            <input
              id="ALB2"
              name="manifestations"
              type="checkbox"
              value="ALB2"
              onChange={handleManifestationsChange}
              checked={manifestations.includes("ALB2")}
              className="form-checkbox h-5 w-5 text-gray-600"
            />
            <span className="ml-2 text-gray-600">Macroalbuminuria</span>
          </label>
        </div>
        <div className="grid grid-cols-2 gap-4">
          <label htmlFor="Angina" className="inline-flex items-center mt-2">
            <input
              id="ANGINA"
              name="manifestations"
              type="checkbox"
              value="ANGINA"
              onChange={handleManifestationsChange}
              checked={manifestations.includes("ANGINA")}
              className="form-checkbox h-5 w-5 text-gray-600"
            />
            <span className="ml-2 text-gray-600">Angina</span>
          </label>
          <label htmlFor="Ictus" className="inline-flex items-center mt-2">
            <input
              id="STROKE"
              name="manifestations"
              type="checkbox"
              value="STROKE"
              onChange={handleManifestationsChange}
              checked={manifestations.includes("STROKE")}
              className="form-checkbox h-5 w-5 text-gray-600"
            />
            <span className="ml-2 text-gray-600">Ictus</span>
          </label>
          <label
            htmlFor="Infarto de Miocardio"
            className="inline-flex items-center mt-2"
          >
            <input
              id="MI"
              name="manifestations"
              type="checkbox"
              value="MI"
              onChange={handleManifestationsChange}
              checked={manifestations.includes("MI")}
              className="form-checkbox h-5 w-5 text-gray-600"
            />
            <span className="ml-2 text-gray-600">Infarto de Miocardio</span>
          </label>
          <label
            htmlFor="Fallo Cardíaco"
            className="inline-flex items-center mt-2"
          >
            <input
              id="HF"
              name="manifestations"
              type="checkbox"
              value="HF"
              onChange={handleManifestationsChange}
              checked={manifestations.includes("HF")}
              className="form-checkbox h-5 w-5 text-gray-600"
            />
            <span className="ml-2 text-gray-600">Fallo Cardíaco</span>
          </label>
          <label htmlFor="Neuropatía" className="inline-flex items-center mt-2">
            <input
              id="NEU"
              name="manifestations"
              type="checkbox"
              value="NEU"
              onChange={handleManifestationsChange}
              checked={manifestations.includes("NEU")}
              className="form-checkbox h-5 w-5 text-gray-600"
            />
            <span className="ml-2 text-gray-600">Neuropatía</span>
          </label>
          <label
            htmlFor="Amputación de Extremidades Inferiores"
            className="inline-flex items-center mt-2"
          >
            <input
              id="LEA"
              name="manifestations"
              type="checkbox"
              value="LEA"
              onChange={handleManifestationsChange}
              checked={manifestations.includes("LEA")}
              className="form-checkbox h-5 w-5 text-gray-600"
            />
            <span className="ml-2 text-gray-600">
              Amputación de Extremidades Inferiores
            </span>
          </label>
        </div>
      </div>
      <div className="mt-8 flex flex-col items-center">
        <button
          type="submit"
          className="bg-gray-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
        >
          Enviar
        </button>
      </div>
    </form>
  );
}
