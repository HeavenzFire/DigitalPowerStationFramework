import Fastify from 'fastify';
import cors from '@fastify/cors';
import fastifyStatic from '@fastify/static';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { createSimulationManager } from './sim/manager.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

async function buildServer() {
  const app = Fastify({ logger: true });

  await app.register(cors, { origin: true });

  // Static files for dashboard
  const publicDir = join(__dirname, '..', 'public');
  await app.register(fastifyStatic, { root: publicDir });

  const sim = createSimulationManager();

  app.get('/api/health', async () => ({ status: 'ok' }));

  app.get('/api/telemetry', async () => sim.getTelemetry());

  app.post<{ Body: { command: string; value?: number } }>(
    '/api/command',
    async (request) => {
      const { command, value } = request.body;
      return sim.handleCommand(command, value);
    }
  );

  // Fallback to index.html
  app.setNotFoundHandler((req, reply) => {
    if (req.raw.url && !req.raw.url.startsWith('/api/')) {
      reply.sendFile('index.html');
      return;
    }
    reply.code(404).send({ error: 'Not found' });
  });

  return app;
}

const PORT = Number(process.env.PORT || 3000);

buildServer()
  .then((app) => app.listen({ port: PORT, host: '0.0.0.0' }))
  .catch((err) => {
    console.error(err);
    process.exit(1);
  });
